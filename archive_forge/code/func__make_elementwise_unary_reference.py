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
def _make_elementwise_unary_reference(type_promotion_kind, *, aten_op=infer_aten_op, extra_meta=None) -> Callable:

    def inner(prim: Callable):
        nonlocal aten_op

        @wraps(prim)
        @out_wrapper()
        @elementwise_unary_scalar_wrapper
        @elementwise_type_promotion_wrapper(type_promoting_args=('a',), type_promotion_kind=type_promotion_kind)
        def _ref(a: TensorLikeType) -> TensorLikeType:
            if extra_meta is not None:
                extra_meta(a)
            output = prim(a)
            return handle_noncontiguous_outputs([a], output)
        if aten_op is infer_aten_op:
            aten_op = utils.get_aten_op(prim, prim.__name__)
        if aten_op is not None:
            register_decomposition(aten_op)(_ref)
        return _ref
    return inner