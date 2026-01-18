from __future__ import annotations
import builtins
import itertools
import operator
from typing import Optional, Sequence
import torch
from . import _dtypes_impl, _util
from ._normalizations import (
from torch import broadcast_shapes
def _concat_check(tup, dtype, out):
    if tup == ():
        raise ValueError('need at least one array to concatenate')
    'Check inputs in concatenate et al.'
    if out is not None and dtype is not None:
        raise TypeError('concatenate() only takes `out` or `dtype` as an argument, but both were provided.')