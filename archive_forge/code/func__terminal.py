from typing import Optional
import numpy as np
from numpy import cos, pi, sin
from gym import core, logger, spaces
from gym.error import DependencyNotInstalled
from gym.envs.classic_control import utils
def _terminal(self):
    s = self.state
    assert s is not None, 'Call reset before using AcrobotEnv object.'
    return bool(-cos(s[0]) - cos(s[1] + s[0]) > 1.0)