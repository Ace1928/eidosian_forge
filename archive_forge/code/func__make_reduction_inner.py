import functools
import itertools
import logging
import os
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import sympy
import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._higher_order_ops.triton_kernel_wrap import (
from torch._prims_common import (
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.utils._sympy.functions import CeilDiv, FloorDiv, ModularIndexing
from .._dynamo.utils import import_submodule
from . import config, inductor_prims, ir, test_operators  # NOQA: F401
from .decomposition import decompositions, get_decompositions
from .ir import (
from .utils import (
from .virtualized import ops, V
from . import kernel
import_submodule(kernel)
from . import quantized_lowerings
def _make_reduction_inner(x, *, axis, keepdims, dtype, override_return_dtype):
    if dtype is not None:
        x = to_dtype(x, dtype)
    size = x.get_size()
    axis = set(_validate_reduction_axis(x, axis))
    kept_sizes = []
    kept_idx = []
    reduced_sizes = []
    reduced_idx = []
    for i in range(len(size)):
        if i in axis:
            reduced_idx.append(i)
            reduced_sizes.append(size[i])
        else:
            kept_idx.append(i)
            kept_sizes.append(size[i])

    def loader(index, reduction_index):
        assert len(reduction_index) == len(reduced_idx)
        if keepdims:
            assert len(index) == len(size)
            index = [index[i] for i in kept_idx]
        assert len(index) == len(kept_idx)
        new_index = [None] * (len(index) + len(reduction_index))
        for idx, var in itertools.chain(zip(kept_idx, index), zip(reduced_idx, reduction_index)):
            new_index[idx] = var
        return inner_loader(new_index)
    if keepdims:
        new_size = list(size)
        for i in reduced_idx:
            new_size[i] = sympy.Integer(1)
    else:
        new_size = kept_sizes
    inner_loader = x.make_loader()
    return dict(device=x.get_device(), dst_dtype=override_return_dtype or x.get_dtype(), src_dtype=x.get_dtype(), inner_fn=loader, ranges=new_size, reduction_ranges=reduced_sizes)