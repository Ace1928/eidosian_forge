from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def _str_to_eviction_policy(eviction_policy):
    eviction = ir.EVICTION_POLICY.NORMAL
    if eviction_policy:
        if eviction_policy == 'evict_last':
            eviction = ir.EVICTION_POLICY.EVICT_LAST
        elif eviction_policy == 'evict_first':
            eviction = ir.EVICTION_POLICY.EVICT_FIRST
        else:
            raise ValueError(f'Eviction policy {eviction_policy} not supported')
    return eviction