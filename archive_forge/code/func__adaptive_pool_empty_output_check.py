import math
from enum import Enum
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._prims_common as utils
from torch import SymBool, SymFloat, Tensor
from torch._decomp import (
from torch._ops import OpOverload
from torch._prims import _prim_elementwise_meta, ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _broadcast_shapes, _maybe_broadcast
from torch.utils import _pytree as pytree
import torch._refs
import torch._refs.nn.functional
import torch._refs.special
def _adaptive_pool_empty_output_check(grad_output: Tensor, arg_name: str):
    ndim = grad_output.ndim
    for i in range(1, ndim):
        torch._check(grad_output.size(i) > 0, lambda: f'{arg_name}(): Expected grad_output to have non-zero size for non-batch dimensions, but grad_output has sizes {grad_output.shape} with dimension {i} being empty')