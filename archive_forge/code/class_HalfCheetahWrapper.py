import numpy as np
import gymnasium as gym
class HalfCheetahWrapper(HalfCheetahEnv or object):
    """Wrapper for the MuJoCo HalfCheetah-v2 environment.

    Adds an additional `reward` method for some model-based RL algos (e.g.
    MB-MPO).
    """

    def reward(self, obs, action, obs_next):
        if obs.ndim == 2 and action.ndim == 2:
            assert obs.shape == obs_next.shape
            forward_vel = obs_next[:, 8]
            ctrl_cost = 0.1 * np.sum(np.square(action), axis=1)
            reward = forward_vel - ctrl_cost
            return np.minimum(np.maximum(-1000.0, reward), 1000.0)
        else:
            forward_vel = obs_next[8]
            ctrl_cost = 0.1 * np.square(action).sum()
            reward = forward_vel - ctrl_cost
            return np.minimum(np.maximum(-1000.0, reward), 1000.0)