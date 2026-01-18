import numpy as np
import gymnasium as gym
class CartPoleWrapper(gym.Wrapper):
    """Wrapper for the CartPole-v1 environment.

    Adds an additional `reward` method for some model-based RL algos (e.g.
    MB-MPO).
    """
    _max_episode_steps = 500

    def __init__(self, **kwargs):
        env = gym.make('CartPole-v1', **kwargs)
        gym.Wrapper.__init__(self, env)

    def reward(self, obs, action, obs_next):
        x = obs_next[:, 0]
        theta = obs_next[:, 2]
        rew = 1.0 - ((x < -self.x_threshold) | (x > self.x_threshold) | (theta < -self.theta_threshold_radians) | (theta > self.theta_threshold_radians)).astype(np.float32)
        return rew