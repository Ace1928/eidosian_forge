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
def _check_spaces(self):
    self._assert_is_running()
    spaces = (self.single_observation_space, self.single_action_space)
    for pipe in self.parent_pipes:
        pipe.send(('_check_spaces', spaces))
    results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
    self._raise_if_errors(successes)
    same_observation_spaces, same_action_spaces = zip(*results)
    if not all(same_observation_spaces):
        raise RuntimeError(f'Some environments have an observation space different from `{self.single_observation_space}`. In order to batch observations, the observation spaces from all environments must be equal.')
    if not all(same_action_spaces):
        raise RuntimeError(f'Some environments have an action space different from `{self.single_action_space}`. In order to batch actions, the action spaces from all environments must be equal.')