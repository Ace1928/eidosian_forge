import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Tuple
import logging
import random
from ray.rllib.env import MultiAgentEnv
def _get_new_pos(self, pos, direction):
    if direction == 0:
        new_pos = (pos[0] - 1, pos[1])
    elif direction == 1:
        new_pos = (pos[0], pos[1] + 1)
    elif direction == 2:
        new_pos = (pos[0] + 1, pos[1])
    elif direction == 3:
        new_pos = (pos[0], pos[1] - 1)
    if new_pos[0] >= 0 and new_pos[0] < self.x_dim and (new_pos[1] >= 0) and (new_pos[1] < self.y_dim) and (self.map[new_pos[0]][new_pos[1]] != '#'):
        return new_pos
    else:
        return pos