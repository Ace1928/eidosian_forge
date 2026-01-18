import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.utils.annotations import Deprecated, DeveloperAPI
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.rllib.utils.typing import (
from ray.util import log_once
def _set_last_observation(self, agent_id, obs):
    self._agent_to_last_obs[agent_id] = obs