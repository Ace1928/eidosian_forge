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
def _zero_pad_in_place(path, value):
    if exclude_states is True and path[0].startswith('state_in_') or path[0] == SampleBatch.SEQ_LENS:
        return
    if value.dtype == object or value.dtype.type is np.str_:
        f_pad = [None] * length
    else:
        f_pad = np.zeros((length,) + np.shape(value)[1:], dtype=value.dtype)
    f_pad_base = f_base = 0
    for len_ in self[SampleBatch.SEQ_LENS]:
        f_pad[f_pad_base:f_pad_base + len_] = value[f_base:f_base + len_]
        f_pad_base += max_seq_len
        f_base += len_
    assert f_base == len(value), value
    curr = self
    for i, p in enumerate(path):
        if i == len(path) - 1:
            curr[p] = f_pad
        curr = curr[p]