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
def _linalg_broadcast_batch_dims_name(arg1: Tensor, arg2: Tensor, name: Optional[str]) -> Tuple[Tensor, Tensor]:
    if name:
        linearSolveCheckInputs(arg1, arg2, name)
    arg1_expand_size, arg2_expand_size = _linalg_broadcast_batch_dims(arg1, arg2)
    arg1_broadcasted = arg1 if arg1_expand_size == arg1.shape else arg1.expand(arg1_expand_size)
    arg2_broadcasted = arg2 if arg2_expand_size == arg2.shape else arg2.expand(arg2_expand_size)
    return (arg1_broadcasted, arg2_broadcasted)