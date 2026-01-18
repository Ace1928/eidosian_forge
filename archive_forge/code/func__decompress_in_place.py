import collections
from functools import partial
import itertools
import sys
from numbers import Number
from typing import Dict, Iterator, Set, Union
from typing import List, Optional
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.utils.annotations import DeveloperAPI, ExperimentalAPI, PublicAPI
from ray.rllib.utils.compression import pack, unpack, is_compressed
from ray.rllib.utils.deprecation import Deprecated, deprecation_warning
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import (
from ray.util import log_once
def _decompress_in_place(path, value):
    if path[0] not in columns:
        return
    curr = self
    for p in path[:-1]:
        curr = curr[p]
    if is_compressed(value):
        curr[path[-1]] = unpack(value)
    elif len(value) > 0 and is_compressed(value[0]):
        curr[path[-1]] = np.array([unpack(o) for o in value])