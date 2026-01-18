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
def _affine_grid_generator_4d(theta: Tensor, size: List[int], align_corners: bool):
    n, _, h, w = size
    base_grid = _make_base_grid_4d(theta, h, w, align_corners=align_corners)
    grid = (base_grid.view(-1, 3, 1) * theta.mT.unsqueeze(1)).sum(-2)
    return grid.view(n, h, w, 2)