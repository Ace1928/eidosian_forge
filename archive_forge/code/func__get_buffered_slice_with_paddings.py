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
def _get_buffered_slice_with_paddings(d, inds):
    element_at_t = []
    for index in inds:
        if index < len(d):
            element_at_t.append(d[index])
        else:
            element_at_t.append(tree.map_structure(np.zeros_like, d[-1]))
    return element_at_t