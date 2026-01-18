import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from ray.rllib.utils.framework import try_import_tf
class CartPoleDebug(CartPoleEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        low = np.concatenate([np.array([0.0]), self.observation_space.low])
        high = np.concatenate([np.array([1000.0]), self.observation_space.high])
        self.observation_space = gym.spaces.Box(low, high, shape=(5,), dtype=np.float32)
        self.timesteps_ = 0
        self._next_action = 0
        self._seed = 1

    def reset(self, *, seed=None, options=None):
        ret = super().reset(seed=self._seed)
        self._seed += 1
        self.timesteps_ = 0
        self._next_action = 0
        obs = np.concatenate([np.array([self.timesteps_]), ret[0]])
        return (obs, ret[1])

    def step(self, action):
        ret = super().step(self._next_action)
        self.timesteps_ += 1
        self._next_action = 0 if self._next_action else 1
        obs = np.concatenate([np.array([self.timesteps_]), ret[0]])
        reward = 0.1 * self.timesteps_
        return (obs, reward) + ret[2:]