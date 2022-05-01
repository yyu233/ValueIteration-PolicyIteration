#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 09:12:40 2020

@author: sonia, parth
"""



from gridWorld import *

from nextState import nextState
from smallGrid import smallGrid 
from mediumGrid import mediumGrid
from testGrid import testGrid
from costFunction import getCost, getCostBridge
import numpy as np
import copy


"""
Implement your value iteration algorithm
"""
uSet = [(1,0),(0,1),(-1,0),(0,-1)]

def valueIteration(gamma,cost,eta,gridname):
    """
    Implement value iteration with a discount factor gamma and 
    the pre-defined cost functions in this assignment. It is passed as a
    string argument above. 
    Output:
    values: Numpy array of (n,m) dimensions
    policy: Numpy array of (n,m) dimensions
    """
    error = 1e-3

    if cost == "cost":
        cost_func = getCost
    else:
        cost_func = getCostBridge

    if gridname == "small":
        n, m, O, START, WINSTATE, LOSESTATE = smallGrid()
    if gridname == "medium":
        n, m, O, START, DISTANTEXIT, CLOSEEXIT, LOSESTATES = mediumGrid()

    policy = [[]]
    values = np.zeros((n, m))
    iterations = 0
    delta = error + 1

    while delta >= error:
        iterations = iterations + 1
        delta = 0
        for i in range(n):
            for j in range(m):
                value_action_map = {}
                tmp = values[i][j]
                #up
                real_next_state_x = i + 0
                real_next_state_y = j + 1
                #next_state = nextState((i, j), (0, 1), eta, O)
                if real_next_state_x < 0 or real_next_state_x >= n or real_next_state_y < 0 or real_next_state_y >= m: #boundary of grid
                    #next_state = (i, j)
                    real_next_state_x = i
                    real_next_state_y = j
                trans_prob = 1 - eta
                if collisionCheck((i, j), (0, 1), O): # up is obstacle
                    #check left
                    if collisionCheck((i ,j), (-1, 0), O):
                        trans_prob = trans_prob + (eta /2)
                    #check right
                    if collisionCheck((i, j), (1, 0), O):
                        trans_prob = trans_prob + (eta /2)
                rewards = trans_prob * cost_func([i, j], gridname)
                up_next_value = rewards + gamma * trans_prob * values[real_next_state_x][real_next_state_y]
                max_next_value = up_next_value
                value_action_map[up_next_value] = (0, 1)

                #left
                real_next_state_x = i + -1
                real_next_state_y = j + 0
                #next_state = nextState((i, j), (-1, 0), eta, O)
                if real_next_state_x < 0 or real_next_state_x >= n or real_next_state_y < 0 or real_next_state_y >= m:#boundary of grid
                    #next_state = (i, j)
                    real_next_state_x = i
                    real_next_state_y = j
                trans_prob = eta / 2
                if collisionCheck((i, j), (-1, 0), O): # left is obstacle
                    #check up
                    if collisionCheck((i ,j), (0, 1), O):
                        trans_prob = trans_prob + (1 - eta)
                    #check right
                    if collisionCheck((i, j), (1, 0), O):
                        trans_prob = trans_prob + (eta /2)
                rewards = trans_prob * cost_func([i, j], gridname)
                left_next_value = rewards + gamma * trans_prob * values[real_next_state_x][real_next_state_y]
                max_next_value = max(max_next_value, left_next_value)
                value_action_map[left_next_value] = (-1, 0)

                #right
                real_next_state_x = i + 1
                real_next_state_y = j + 0
                #next_state = nextState((i, j), (1, 0), eta, O)
                if real_next_state_x < 0 or real_next_state_x >= n or real_next_state_y < 0 or real_next_state_y >= m:#boundary of grid
                    #next_state = (i, j)
                    real_next_state_x = i
                    real_next_state_y = j
                trans_prob = eta / 2
                if collisionCheck((i, j), (1, 0), O): # right is obstacle
                    #check up
                    if collisionCheck((i ,j), (0, 1), O):
                        trans_prob = trans_prob + (1 - eta)
                    #check left
                    if collisionCheck((i, j), (-1, 0), O):
                        trans_prob = trans_prob + (eta /2)
                rewards = trans_prob * cost_func([i, j], gridname)
                right_next_value = rewards + gamma * trans_prob * values[real_next_state_x][real_next_state_y]
                max_next_value = max(max_next_value, right_next_value)
                value_action_map[right_next_value] = (1, 0)
                
                #down
                real_next_state_x = i + 0
                real_next_state_y = j - 1
                #next_state = nextState((i, j), (0, -1), eta, O)
                if real_next_state_x < 0 or real_next_state_x >= n or real_next_state_y < 0 or real_next_state_y >= m:#boundary of grid
                    #next_state = (i, j)
                    real_next_state_x = i
                    real_next_state_y = j
                rewards = (0) * cost_func([i, j], gridname)
                down_next_value = rewards + gamma * (0) * values[real_next_state_x][real_next_state_y]
                max_next_value = max(max_next_value, down_next_value)
                value_action_map[down_next_value] = (0, -1)

                values[i][j] = max_next_value
                delta = max(delta, abs(tmp - values[i][j]))
                #optimal policy
                policy[i].append(value_action_map[max_next_value])
            if len(policy) < n:
                policy.append([])
        #print(delta)

    
    return values, policy, iterations

"""
Implement your policy iteration algorithm
"""

def policyIteration(gamma,cost,eta,gridname):
    """
    (Offline) Policy iteration with a discount factor gamma and 
    pre-defined cost functions. 
    Output:
    values: Numpy array of (n,m) dimensions
    policy: Numpy array of (n,m) dimensions
    """
    error = 1e-3

    if cost == "cost":
        cost_func = getCost
    else:
        cost_func = getCostBridge

    if gridname == "small":
        n, m, O, START, WINSTATE, LOSESTATE = smallGrid()
    if gridname == "medium":
        n, m, O, START, DISTANTEXIT, CLOSEEXIT, LOSESTATES = mediumGrid()

    policy = [[(0, 1)] * m] * n
    values = np.zeros((n, m))
    iterations = 0
    delta = error + 1

    while delta >= error:
        iterations = iterations + 1
        delta = 0
        for i in range(n):
            for j in range(m):
                tmp = values[i][j]
                if policy[i][j] == (0, 1):
                    #up
                    real_next_state_x = i + 0
                    real_next_state_y = j + 1
                    if real_next_state_x < 0 or real_next_state_x >= n or real_next_state_y < 0 or real_next_state_y >= m: #boundary of grid
                        real_next_state_x = i
                        real_next_state_y = j
                    trans_prob = 1 - eta
                    if collisionCheck((i, j), (0, 1), O): # up is obstacle
                        #check left
                        if collisionCheck((i ,j), (-1, 0), O):
                            trans_prob = trans_prob + (eta /2)
                        #check right
                        if collisionCheck((i, j), (1, 0), O):
                            trans_prob = trans_prob + (eta /2)
                    rewards = trans_prob * cost_func([i, j], gridname)
                    up_next_value = rewards + gamma * trans_prob * values[real_next_state_x][real_next_state_y]
                    values[i][j] = up_next_value

                if policy[i][j] == (-1, 0):
                    #left
                    real_next_state_x = i + -1
                    real_next_state_y = j + 0
                    if real_next_state_x < 0 or real_next_state_x >= n or real_next_state_y < 0 or real_next_state_y >= m:#boundary of grid
                        real_next_state_x = i
                        real_next_state_y = j
                    trans_prob = eta / 2
                    if collisionCheck((i, j), (-1, 0), O): # left is obstacle
                        #check up
                        if collisionCheck((i ,j), (0, 1), O):
                            trans_prob = trans_prob + (1 - eta)
                        #check right
                        if collisionCheck((i, j), (1, 0), O):
                            trans_prob = trans_prob + (eta /2)
                    rewards = trans_prob * cost_func([i, j], gridname)
                    left_next_value = rewards + gamma * trans_prob * values[real_next_state_x][real_next_state_y]
                    values[i][j] = left_next_value

                if policy[i][j] == (1, 0):
                    #right
                    real_next_state_x = i + 1
                    real_next_state_y = j + 0
                    if real_next_state_x < 0 or real_next_state_x >= n or real_next_state_y < 0 or real_next_state_y >= m:#boundary of grid
                        real_next_state_x = i
                        real_next_state_y = j
                    trans_prob = eta / 2
                    if collisionCheck((i, j), (1, 0), O): # right is obstacle
                        #check up
                        if collisionCheck((i ,j), (0, 1), O):
                            trans_prob = trans_prob + (1 - eta)
                        #check left
                        if collisionCheck((i, j), (-1, 0), O):
                            trans_prob = trans_prob + (eta /2)
                    rewards = trans_prob * cost_func([i, j], gridname)
                    right_next_value = rewards + gamma * trans_prob * values[real_next_state_x][real_next_state_y]
                    values[i][j] = right_next_value

                if policy[i][j] == (0, -1):
                    #down
                    real_next_state_x = i + 0
                    real_next_state_y = j - 1
                    if real_next_state_x < 0 or real_next_state_x >= n or real_next_state_y < 0 or real_next_state_y >= m:#boundary of grid
                        real_next_state_x = i
                        real_next_state_y = j
                    rewards = (0) * cost_func([i, j], gridname)
                    down_next_value = rewards + gamma * (0) * values[real_next_state_x][real_next_state_y]
                    values[i][j] = down_next_value

                delta = max(delta, abs(tmp - values[i][j]))

    #policy  improvement
    for i in range(n):
        for j in range(m):
                value_action_map = {}
                tmp = values[i][j]
                #up
                real_next_state_x = i + 0
                real_next_state_y = j + 1
                if real_next_state_x < 0 or real_next_state_x >= n or real_next_state_y < 0 or real_next_state_y >= m: #boundary of grid
                    real_next_state_x = i
                    real_next_state_y = j
                trans_prob = 1 - eta
                if collisionCheck((i, j), (0, 1), O): # up is obstacle
                    #check left
                    if collisionCheck((i ,j), (-1, 0), O):
                        trans_prob = trans_prob + (eta /2)
                    #check right
                    if collisionCheck((i, j), (1, 0), O):
                        trans_prob = trans_prob + (eta /2)
                rewards = trans_prob * cost_func([i, j], gridname)
                up_next_value = rewards + gamma * trans_prob * values[real_next_state_x][real_next_state_y]
                
                if up_next_value > tmp and policy[i][j] != (0, 1):
                    policy[i][j] = (0, 1)
                    tmp = up_next_value
                
                #left
                real_next_state_x = i + -1
                real_next_state_y = j + 0
                if real_next_state_x < 0 or real_next_state_x >= n or real_next_state_y < 0 or real_next_state_y >= m:#boundary of grid
                    real_next_state_x = i
                    real_next_state_y = j
                trans_prob = eta / 2
                if collisionCheck((i, j), (-1, 0), O): # left is obstacle
                    #check up
                    if collisionCheck((i ,j), (0, 1), O):
                        trans_prob = trans_prob + (1 - eta)
                    #check right
                    if collisionCheck((i, j), (1, 0), O):
                        trans_prob = trans_prob + (eta /2)
                rewards = trans_prob * cost_func([i, j], gridname)
                left_next_value = rewards + gamma * trans_prob * values[real_next_state_x][real_next_state_y]
                
                if left_next_value > tmp and policy[i][j] != (-1, 0):
                    policy[i][j] = (-1, 0)
                    tmp = left_next_value   
                
                #right
                real_next_state_x = i + 1
                real_next_state_y = j + 0
                if real_next_state_x < 0 or real_next_state_x >= n or real_next_state_y < 0 or real_next_state_y >= m:#boundary of grid
                    real_next_state_x = i
                    real_next_state_y = j
                trans_prob = eta / 2
                if collisionCheck((i, j), (1, 0), O): # right is obstacle
                    #check up
                    if collisionCheck((i ,j), (0, 1), O):
                        trans_prob = trans_prob + (1 - eta)
                    #check left
                    if collisionCheck((i, j), (-1, 0), O):
                        trans_prob = trans_prob + (eta /2)
                rewards = trans_prob * cost_func([i, j], gridname)
                right_next_value = rewards + gamma * trans_prob * values[real_next_state_x][real_next_state_y]
                
                if right_next_value > tmp and policy[i][j] != (1, 0):
                    policy[i][j] = (1, 0)
                    tmp = right_next_value

                #down
                real_next_state_x = i + 0
                real_next_state_y = j - 1
                if real_next_state_x < 0 or real_next_state_x >= n or real_next_state_y < 0 or real_next_state_y >= m:#boundary of grid
                    real_next_state_x = i
                    real_next_state_y = j
                rewards = (0) * cost_func([i, j], gridname)
                down_next_value = rewards + gamma * (0) * values[real_next_state_x][real_next_state_y]

                if down_next_value > tmp and policy[i][j] != (0, -1):
                    policy[i][j] = (0, -1)
                    tmp = down_next_value

    
    return values, policy, iterations

def optimalValues(question):
    """
    Please input your values of gamma and eta
    for each assignment problem here.
    """
    if question=='a':
        gamma=0.9
        eta=0.2
        return gamma, eta
    elif question=='b':
        gamma=0.9
        eta=0.2
        return gamma, eta
    elif question=='c':
        gamma=0.9
        eta=0.2
        return gamma, eta
    elif question=='d1':
        gamma=0.9
        eta=0.2
        return gamma, eta
    elif question=='d2':
        gamma=0.9
        eta=0.2
        return gamma, eta
    elif question=='d3':
        gamma=0.9
        eta=0.2
        return gamma, eta
    elif question=='d4':
        gamma=0.9
        eta=0.2
        return gamma, eta
    else: 
        pass
    return 0
    
def showPath(xI,xG,path,n,m,O):
    gridpath = makePath(xI,xG,path,n,m,O)
    fig, ax = plt.subplots(1, 1) # make a figure + axes
    ax.imshow(gridpath) # Plot it
    ax.invert_yaxis() # Needed so that bottom left is (0,0)
    
    
# Function to actually plot the cost-to-gos
def plotValues(values,xI,xG,n,m,O):
    gridvalues = makeValues(values,xI,xG,n,m,O)
    fig, ax = plt.subplots() # make a figure + axes
    ax.imshow(gridvalues) # Plot it
    ax.invert_yaxis() # Needed so that bottom left is (0,0)


def showValues(n,m,values,O):
    string = '------'
    for i in range(0, n):
        string = string + '-----'
    for j in range(0, m):
        print(string)
        out = '| '
        for i in range(0, n):            
            jind = m-j-1 # Need to reverse index so bottom-left is (0,0)
            if isObstacle((i,jind),O):
                out += 'Obs' + ' | '
            else:
                out += str(values[i,jind]) + ' | '
        print(out)
    print(string)    


if __name__ == '__main__':
    gridname='small'
    # Use small and medium grid for your code submission
    # cost types: {'cost', 'bridge'}
    if gridname=='small':
        n, m, O, START, WINSTATE, LOSESTATE = smallGrid()
    elif gridname=='medium':
        n, m, O, START, DISTANTEXIT, CLOSEEXIT, LOSESTATES = mediumGrid()
    # elif gridname=='test':
    #     n, m, O, START, WINSTATE, DISTANTEXIT, LOSESTATE, LOSESTATES = testGrid()
    else:
        raise NameError("Unknown grid")
    
    # values, policy = valueIteration()
    # values, policy = policyIteration()
    """
    # Case 1:
    """
    print("########## Case 1 ##########")
    #gridname = 'medium'
    gridname = 'small'
    values, policy, iterations = valueIteration(0.9, "cost", 0.2, "small")
    #print(f"values: {values}")
    #print(f"policy: {policy}")
    print(f"gamma: {0.9}, eta: {0.2}, iterations: {iterations}")
    
    values, policy, iterations = valueIteration(0.5, "cost", 0.2, "small")
    #print(f"values: {values}")
    #print(f"policy: {policy}")
    print(f"gamma: {0.5}, eta: {0.2}, iterations: {iterations}")
    # values, policy = policyIteration()

    values, policy, iterations = valueIteration(0.1, "cost", 0.2, "small")
    #print(f"values: {values}")
    #print(f"policy: {policy}")
    print(f"gamma: {0.1}, eta: {0.2}, iterations: {iterations}")



    values, policy, iterations = valueIteration(0.9, "cost", 0.5, "small")
    #print(f"values: {values}")
    #print(f"policy: {policy}")
    print(f"gamma: {0.9}, eta: {0.5}, iterations: {iterations}")
    
    values, policy, iterations = valueIteration(0.5, "cost", 0.5, "small")
    #print(f"values: {values}")
    #print(f"policy: {policy}")
    print(f"gamma: {0.5}, eta: {0.5}, iterations: {iterations}")
    # values, policy = policyIteration()

    values, policy, iterations = valueIteration(0.1, "cost", 0.5, "small")
    #print(f"values: {values}")
    #print(f"policy: {policy}")
    print(f"gamma: {0.1}, eta: {0.5}, iterations: {iterations}")



    values, policy, iterations = valueIteration(0.9, "cost", 0.9, "small")
    #print(f"values: {values}")
    #print(f"policy: {policy}")
    print(f"gamma: {0.9}, eta: {0.9}, iterations: {iterations}")
    
    values, policy, iterations = valueIteration(0.5, "cost", 0.9, "small")
    #print(f"values: {values}")
    #print(f"policy: {policy}")
    print(f"gamma: {0.5}, eta: {0.9}, iterations: {iterations}")
    # values, policy = policyIteration()

    values, policy, iterations = valueIteration(0.1, "cost", 0.9, "small")
    #print(f"values: {values}")
    #print(f"policy: {policy}")
    print(f"gamma: {0.1}, eta: {0.9}, iterations: {iterations}")
    #values, policy, iterations = valueIteration(0.9, "cost", 0.8, "small")

    """
    #Case 2
    """
    print("########## Case 2 ##########")
    values, policy, iterations = policyIteration(0.9, "cost", 0.2, "small")
    print(f"gamma: {0.9}, eta: {0.2}, iterations: {iterations}")

    """
    # Case 3
    """
    """
    # Case 4
    """
    
    # # Sample use of plotValues from gridWorld
    # values = np.zeros((n,m))
    # # Loop through values to just assign some dummy/arbitrary data
    # for i in range(n):
    #     for j in range(m):
    #         if not(isObstacle((i,j),O)):
    #             values[i][j] = (n+2*m-2) - (i + 2*j)
    # # This will print those numeric values as console text
    # showValues(n,m,values,O)

    # # This will plot the actual grid with objects as black and values as
    # # shades from green to red in increasing numerical order
    # xI = [1,1]
    # xG = [4,3]
    # grid = create_binary_grid(n, m, O)
    # plotValues(grid*values,xI,xG,n,m,O)
    # path = [[2,1],[3,1],[4,1],[4,2],[4,3]]
    # showPath(xI,xG,path,n,m,O)

    

    # plt.show()
