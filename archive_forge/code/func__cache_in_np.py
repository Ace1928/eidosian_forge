import copy
import logging
import math
from typing import Any, Dict, List, Optional
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Space
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.spaces.space_utils import (
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
def _cache_in_np(self, cache_dict: Dict[str, List[np.ndarray]], key: str) -> None:
    """Caches the numpy version of the key in the buffer dict."""
    if key not in cache_dict:
        cache_dict[key] = [_to_float_np_array(d) for d in self.buffers[key]]