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
def _adaptive_pooling_idx_sum(kernel_maxes, start_index_fns, end_index_fns):
    h_start_index_fn, w_start_index_fn = start_index_fns
    h_end_index_fn, w_end_index_fn = end_index_fns

    def fn_sum(idx, loader):
        *prefix, bh, bw = idx
        h_start_index = h_start_index_fn(bh)
        h_end_index = h_end_index_fn(bh)
        w_start_index = w_start_index_fn(bw)
        w_end_index = w_end_index_fn(bw)
        total = None
        for ih, iw in itertools.product(range(kernel_maxes[0]), range(kernel_maxes[1])):
            val = loader(prefix, [ih, iw], [h_start_index, w_start_index], [h_end_index, w_end_index])
            if total is None:
                total = val
            else:
                total = ops.add(val, total)
        return total
    return fn_sum