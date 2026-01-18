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
@property
def _policy_mapping_fn(self):
    deprecation_warning(old='Episode._policy_mapping_fn', new='Episode.policy_mapping_fn', error=True)
    return self.policy_mapping_fn