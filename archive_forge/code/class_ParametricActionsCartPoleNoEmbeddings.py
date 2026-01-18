import random
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
class ParametricActionsCartPoleNoEmbeddings(gym.Env):
    """Same as the above ParametricActionsCartPole.

    However, action embeddings are not published inside observations,
    but will be learnt by the model.

    At each step, we emit a dict of:
        - the actual cart observation
        - a mask of valid actions (e.g., [0, 0, 1, 0, 0, 1] for 6 max avail)
        - action embeddings (w/ "dummy embedding" for invalid actions) are
          outsourced in the model and will be learned.
    """

    def __init__(self, max_avail_actions):
        self.left_idx, self.right_idx = random.sample(range(max_avail_actions), 2)
        self.valid_avail_actions_mask = np.array([0.0] * max_avail_actions, dtype=np.int8)
        self.valid_avail_actions_mask[self.left_idx] = 1
        self.valid_avail_actions_mask[self.right_idx] = 1
        self.action_space = Discrete(max_avail_actions)
        self.wrapped = gym.make('CartPole-v1')
        self.observation_space = Dict({'valid_avail_actions_mask': Box(0, 1, shape=(max_avail_actions,)), 'cart': self.wrapped.observation_space})

    def reset(self, *, seed=None, options=None):
        obs, infos = self.wrapped.reset()
        return ({'valid_avail_actions_mask': self.valid_avail_actions_mask, 'cart': obs}, infos)

    def step(self, action):
        if action == self.left_idx:
            actual_action = 0
        elif action == self.right_idx:
            actual_action = 1
        else:
            raise ValueError('Chosen action was not one of the non-zero action embeddings', action, self.valid_avail_actions_mask, self.left_idx, self.right_idx)
        orig_obs, rew, done, truncated, info = self.wrapped.step(actual_action)
        obs = {'valid_avail_actions_mask': self.valid_avail_actions_mask, 'cart': orig_obs}
        return (obs, rew, done, truncated, info)