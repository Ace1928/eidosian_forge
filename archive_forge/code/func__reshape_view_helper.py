import builtins
import collections
import inspect
import itertools
import math
import operator
import warnings
from collections.abc import Iterable
from enum import Enum
from functools import partial, reduce, singledispatch, wraps
from typing import Any, Callable, Dict, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch import sym_float, sym_int
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._decomp import register_decomposition
import torch._refs._conversions
import torch._refs.fft
import torch._refs.linalg
import torch._refs.nn.functional
import torch._refs.special
def _reshape_view_helper(a: TensorLikeType, *shape, allow_copy: bool) -> TensorLikeType:
    shape = utils.extract_shape_from_varargs(shape, validate=False)
    shape = utils.infer_size(shape, a.numel())
    if tuple(a.shape) == tuple(shape):
        return prims.view_of(a)
    if a.numel() == 0:
        return as_strided(a, shape, utils.make_contiguous_strides_for(shape))
    if a.ndim == 0:
        _a = a
        for length in shape:
            assert length == 1
            _a = unsqueeze(_a, -1)
        return _a
    if len(shape) == 0:
        _a = a
        for length in a.shape:
            assert length == 1
            _a = squeeze(_a, -1)
        return _a
    idx = 0
    a_ = a
    for length in shape:
        if idx >= a_.ndim:
            assert length == 1
            last_dim = a_.ndim - 1
            a_ = prims.split_dim(a_, last_dim, a_.shape[last_dim])
            idx = idx + 1
            continue
        if length == a_.shape[idx]:
            idx = idx + 1
            continue
        accum = a_.shape[idx]
        end = idx
        while accum % length != 0:
            end = end + 1
            accum = accum * a_.shape[end]
        if end != idx:
            new_shape, new_strides = prims._collapse_view_helper(a_, idx, end)
            if new_shape is None:
                if allow_copy:
                    return prims.reshape(a, shape)
                msg = 'Cannot view a tensor with shape {} and strides {} as a tensor with shape {}!'.format(a.shape, a.stride(), shape)
                raise ValueError(msg)
            a_ = flatten(a_, idx, end)
        if accum != length:
            a_ = prims.split_dim(a_, idx, length)
        idx = idx + 1
    while idx < a_.ndim:
        assert a_.shape[idx] == 1
        a_ = squeeze(a_, idx)
    return a_