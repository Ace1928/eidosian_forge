from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Union
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.images import rgb2gray, resize
@PublicAPI
class FrameStackTrajectoryView(gym.ObservationWrapper):

    def __init__(self, env):
        """No stacking. Trajectory View API takes care of this."""
        gym.Wrapper.__init__(self, env)
        shp = env.observation_space.shape
        assert shp[2] == 1
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1]), dtype=env.observation_space.dtype)

    def observation(self, observation):
        return np.squeeze(observation, axis=-1)