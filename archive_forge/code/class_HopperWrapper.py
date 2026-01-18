import numpy as np
import gymnasium as gym
class HopperWrapper(HopperEnv or object):
    """Wrapper for the MuJoCo Hopper-v2 environment.

    Adds an additional `reward` method for some model-based RL algos (e.g.
    MB-MPO).
    """
    _max_episode_steps = 1000

    def reward(self, obs, action, obs_next):
        alive_bonus = 1.0
        assert obs.ndim == 2 and action.ndim == 2
        assert obs.shape == obs_next.shape and action.shape[0] == obs.shape[0]
        vel = obs_next[:, 5]
        ctrl_cost = 0.001 * np.sum(np.square(action), axis=1)
        reward = vel + alive_bonus - ctrl_cost
        return np.minimum(np.maximum(-1000.0, reward), 1000.0)