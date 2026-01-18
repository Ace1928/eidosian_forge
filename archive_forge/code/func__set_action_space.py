from os import path
from typing import Optional, Union
import numpy as np
import gym
from gym import error, logger, spaces
from gym.spaces import Space
def _set_action_space(self):
    bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
    low, high = bounds.T
    self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
    return self.action_space