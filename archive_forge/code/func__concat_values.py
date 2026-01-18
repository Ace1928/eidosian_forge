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
def _concat_values(*values, time_major=None) -> TensorType:
    """Concatenates a list of values.

    Args:
        values: The values to concatenate.
        time_major: Whether to concatenate along the first axis
            (time_major=False) or the second axis (time_major=True).
    """
    if torch and torch.is_tensor(values[0]):
        return torch.cat(values, dim=1 if time_major else 0)
    elif isinstance(values[0], np.ndarray):
        return np.concatenate(values, axis=1 if time_major else 0)
    elif tf and tf.is_tensor(values[0]):
        return tf.concat(values, axis=1 if time_major else 0)
    elif isinstance(values[0], list):
        concatenated_list = []
        for sublist in values:
            concatenated_list.extend(sublist)
        return concatenated_list
    else:
        raise ValueError(f'Unsupported type for concatenation: {type(values[0])} first element: {values[0]}')