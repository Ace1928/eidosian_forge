import gymnasium as gym
import numpy as np
class OneHot(gym.Wrapper):

    def __init__(self, env):
        super(OneHot, self).__init__(env)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n,))

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return (self._encode_obs(obs), info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (self._encode_obs(obs), reward, terminated, truncated, info)

    def _encode_obs(self, obs):
        new_obs = np.ones(self.env.observation_space.n)
        new_obs[obs] = 1.0
        return new_obs