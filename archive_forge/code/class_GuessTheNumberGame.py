import gymnasium as gym
import numpy as np
import random
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
from ray.rllib.examples.env.mock_env import MockEnv, MockEnv2
from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole
from ray.rllib.utils.deprecation import Deprecated
class GuessTheNumberGame(MultiAgentEnv):
    """
    We have two players, 0 and 1. Agent 0 has to pick a number between 0, MAX-1
    at reset. Agent 1 has to guess the number by asking N questions of whether
    of the form of "a <number> is higher|lower|equal to the picked number. The
    action space is MultiDiscrete [3, MAX]. For the first index 0 means lower,
    1 means higher and 2 means equal. The environment answers with yes (1) or
    no (0) on the reward function. Every time step that agent 1 wastes agent 0
    gets a reward of 1. After N steps the game is terminated. If agent 1
    guesses the number correctly, it gets a reward of 100 points, otherwise it
    gets a reward of 0. On the other hand if agent 0 wins they win 100 points.
    The optimal policy controlling agent 1 should converge to a binary search
    strategy.
    """
    MAX_NUMBER = 3
    MAX_STEPS = 20

    def __init__(self, config):
        super().__init__()
        self._agent_ids = {0, 1}
        self.max_number = config.get('max_number', self.MAX_NUMBER)
        self.max_steps = config.get('max_steps', self.MAX_STEPS)
        self._number = None
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.MultiDiscrete([3, self.max_number])

    def reset(self, *, seed=None, options=None):
        self._step = 0
        self._number = None
        return ({0: 0}, {})

    def step(self, action_dict):
        agent_0_action = action_dict.get(0)
        if agent_0_action is not None:
            self._number = agent_0_action[1]
            return ({1: 0}, {0: 0}, {0: False, '__all__': False}, {0: False, '__all__': False}, {})
        if self._number is None:
            raise ValueError('No number is selected by agent 0. Have you restarted the environment?')
        direction, number = action_dict.get(1)
        info = {}
        obs = {1: 0}
        guessed_correctly = False
        terminated = {1: False, '__all__': False}
        truncated = {1: False, '__all__': False}
        if direction == 0:
            reward = {1: int(number > self._number), 0: 1}
        elif direction == 1:
            reward = {1: int(number < self._number), 0: 1}
        else:
            guessed_correctly = number == self._number
            reward = {1: guessed_correctly * 100, 0: guessed_correctly * -100}
            terminated = {1: guessed_correctly, '__all__': guessed_correctly}
        self._step += 1
        if self._step >= self.max_steps:
            truncated['__all__'] = True
            if not guessed_correctly:
                reward[0] = 100
        return (obs, reward, terminated, truncated, info)