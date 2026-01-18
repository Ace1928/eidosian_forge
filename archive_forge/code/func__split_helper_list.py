from __future__ import annotations
import builtins
import itertools
import operator
from typing import Optional, Sequence
import torch
from . import _dtypes_impl, _util
from ._normalizations import (
from torch import broadcast_shapes
def _split_helper_list(tensor, indices_or_sections, axis):
    if not isinstance(indices_or_sections, list):
        raise NotImplementedError('split: indices_or_sections: list')
    lst = [x for x in indices_or_sections if x <= tensor.shape[axis]]
    num_extra = len(indices_or_sections) - len(lst)
    lst.append(tensor.shape[axis])
    lst = [lst[0]] + [a - b for a, b in zip(lst[1:], lst[:-1])]
    lst += [0] * num_extra
    return torch.split(tensor, lst, axis)