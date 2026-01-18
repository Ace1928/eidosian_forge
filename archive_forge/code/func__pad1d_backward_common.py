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
def _pad1d_backward_common(grad_output, input, padding, *, is_reflection):
    dim_w = 1
    if not is_reflection:
        torch._check(len(padding) == 2, lambda: 'padding size is expected to be 2')
    if input.ndim == 3:
        dim_w += 1
    pad_l, pad_r = padding
    input_w = input.size(dim_w)
    output_w = input_w + pad_l + pad_r
    if is_reflection:
        torch._check(pad_l < input_w and pad_r < input_w, lambda: f'Argument #4: Padding size should be less than the corresponding input dimension, but got: padding ({pad_l}, {pad_r}) at dimension {dim_w} of input {input.shape}')
    torch._check(output_w == grad_output.size(dim_w), lambda: f'grad_output width unexpected. Expected: {output_w}, Got: {grad_output.size(dim_w)}')
    return input.new_empty(input.shape)