from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def _str_to_store_cache_modifier(cache_modifier):
    cache = ir.CACHE_MODIFIER.NONE
    if cache_modifier:
        if cache_modifier == '.wb':
            cache = ir.CACHE_MODIFIER.WB
        elif cache_modifier == '.cg':
            cache = ir.CACHE_MODIFIER.CG
        elif cache_modifier == '.cs':
            cache = ir.CACHE_MODIFIER.CS
        elif cache_modifier == '.wt':
            cache = ir.CACHE_MODIFIER.WT
        else:
            raise ValueError(f'Cache modifier {cache_modifier} not supported')
    return cache