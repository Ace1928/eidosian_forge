import functools
import numbers
import operator
import sys
from enum import Enum
from functools import partial, reduce
from itertools import chain, product
from typing import Callable, cast, Iterable, List, Optional, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch.nn.functional as F
from torch import sym_float, sym_int, Tensor
from torch._decomp import register_decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import IntLike, NumberType, TensorLike, TensorSequenceType
from torch._prims_common.wrappers import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
def _make_base_grid_4d(theta: Tensor, h: int, w: int, align_corners: bool):
    dtype = theta.dtype
    device = theta.device
    grid_x = _linspace_from_neg_one(w, align_corners, dtype, device).view(1, w, 1)
    grid_y = _linspace_from_neg_one(h, align_corners, dtype, device).view(h, 1, 1)
    grid_one = torch.ones((1, 1, 1), dtype=dtype, device=device)
    grid_x = torch.nn.functional.pad(grid_x, pad=(0, 2), mode='constant', value=0)
    grid_y = torch.nn.functional.pad(grid_y, pad=(1, 1), mode='constant', value=0)
    grid_one = torch.nn.functional.pad(grid_one, pad=(2, 0), mode='constant', value=0)
    return grid_x + grid_y + grid_one