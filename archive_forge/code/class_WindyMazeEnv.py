import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Tuple
import logging
import random
from ray.rllib.env import MultiAgentEnv
class WindyMazeEnv(gym.Env):

    def __init__(self, env_config):
        self.map = [m for m in MAP_DATA.split('\n') if m]
        self.x_dim = len(self.map)
        self.y_dim = len(self.map[0])
        logger.info('Loaded map {} {}'.format(self.x_dim, self.y_dim))
        for x in range(self.x_dim):
            for y in range(self.y_dim):
                if self.map[x][y] == 'S':
                    self.start_pos = (x, y)
                elif self.map[x][y] == 'F':
                    self.end_pos = (x, y)
        logger.info('Start pos {} end pos {}'.format(self.start_pos, self.end_pos))
        self.observation_space = Tuple([Box(0, 100, shape=(2,)), Discrete(4)])
        self.action_space = Discrete(2)

    def reset(self, *, seed=None, options=None):
        self.wind_direction = random.choice([0, 1, 2, 3])
        self.pos = self.start_pos
        self.num_steps = 0
        return ([[self.pos[0], self.pos[1]], self.wind_direction], {})

    def step(self, action):
        if action == 1:
            self.pos = self._get_new_pos(self.pos, self.wind_direction)
        self.num_steps += 1
        self.wind_direction = random.choice([0, 1, 2, 3])
        at_goal = self.pos == self.end_pos
        truncated = self.num_steps >= 200
        done = at_goal or truncated
        return ([[self.pos[0], self.pos[1]], self.wind_direction], 100 * int(at_goal), done, truncated, {})

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