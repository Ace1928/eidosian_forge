import multiprocessing as mp
import sys
import time
from copy import deepcopy
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import gym
from gym import logger
from gym.core import ObsType
from gym.error import (
from gym.vector.utils import (
from gym.vector.vector_env import VectorEnv
def _assert_is_running(self):
    if self.closed:
        raise ClosedEnvironmentError(f'Trying to operate on `{type(self).__name__}`, after a call to `close()`.')