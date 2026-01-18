import math
from typing import Optional
import numpy as np
import gym
from gym import spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
def _height(self, xs):
    return np.sin(3 * xs) * 0.45 + 0.55