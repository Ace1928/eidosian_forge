import logging
import os
import warnings
from typing import Dict, List, Optional, TYPE_CHECKING, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Discrete, MultiDiscrete
from packaging import version
import ray
from ray.rllib.models.repeated_values import RepeatedValues
from ray.rllib.utils.annotations import Deprecated, PublicAPI, DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import SMALL_NUMBER
from ray.rllib.utils.typing import (
def _dynamo_is_available():
    try:
        import torch._dynamo as dynamo
        return True
    except ImportError:
        return False