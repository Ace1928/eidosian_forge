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
def _padding_check_valid_input(input, padding, *, dim):
    torch._check(len(padding) == 2 * dim, lambda: f'padding size is expected to be {2 * dim}, but got: {len(padding)}')
    input_dim = input.ndim
    is_batch_mode = input_dim == dim + 2
    valid_batch_mode = is_batch_mode
    valid_non_batch_mode = not is_batch_mode
    if is_batch_mode:
        for d in range(1, input_dim):
            valid_batch_mode = valid_batch_mode and input.size(d) != 0
    else:
        for d in range(0, input_dim):
            valid_non_batch_mode = valid_non_batch_mode and input.size(d) != 0
    torch._check(valid_batch_mode or valid_non_batch_mode, lambda: f'Expected {dim + 1}D or {dim + 2}D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: {input.shape}')