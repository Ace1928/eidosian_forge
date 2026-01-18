import gymnasium as gym
from gymnasium import spaces
Modified version of the CliffWalking environment from Farama-Foundation's
    Gymnasium with walls instead of a cliff.

    ### Description
    The board is a 4x12 matrix, with (using NumPy matrix indexing):
    - [3, 0] or obs==36 as the start at bottom-left
    - [3, 11] or obs==47 as the goal at bottom-right
    - [3, 1..10] or obs==37...46 as the cliff at bottom-center

    An episode terminates when the agent reaches the goal.

    ### Actions
    There are 4 discrete deterministic actions:
    - 0: move up
    - 1: move right
    - 2: move down
    - 3: move left
    You can also use the constants ACTION_UP, ACTION_RIGHT, ... defined above.

    ### Observations
    There are 3x12 + 2 possible states, not including the walls. If an action
    would move an agent into one of the walls, it simply stays in the same position.

    ### Reward
    Each time step incurs -1 reward, except reaching the goal which gives +10 reward.
    