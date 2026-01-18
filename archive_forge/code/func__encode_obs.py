import gymnasium as gym
import numpy as np
def _encode_obs(self, obs):
    new_obs = np.ones(self.env.observation_space.n)
    new_obs[obs] = 1.0
    return new_obs