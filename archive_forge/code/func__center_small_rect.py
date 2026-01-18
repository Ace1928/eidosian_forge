from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional
import numpy as np
from gym import Env, logger, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled
@staticmethod
def _center_small_rect(big_rect, small_dims):
    offset_w = (big_rect[2] - small_dims[0]) / 2
    offset_h = (big_rect[3] - small_dims[1]) / 2
    return (big_rect[0] + offset_w, big_rect[1] + offset_h)