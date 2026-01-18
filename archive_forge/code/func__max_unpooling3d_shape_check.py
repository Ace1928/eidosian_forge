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
def _max_unpooling3d_shape_check(input, indices, output_size, stride, padding, fn_name):
    torch._check(indices.dtype == torch.int64, lambda: 'elements in indices should be type int64')
    torch._check(input.ndim in (4, 5), lambda: f'Input to max_unpooling3d should be a 4d or 5d Tensor, but got a tensor with {input.ndim} dimensions.')
    torch._check(len(output_size) == 3, lambda: f'There should be exactly three elements (depth, height, width) in output_size, but got {len(output_size)} elements.')
    torch._check(len(stride) == 3, lambda: f'There should be exactly three elements (depth, height, width) in stride, but got: {len(stride)} elements.')
    torch._check(len(padding) == 3, lambda: f'There should be exactly three elements (depth, height, width) in padding, but got: {len(padding)} elements.')
    torch._check(input.shape == indices.shape, lambda: f'Expected shape of indices to be same as that of the input tensor ({input.shape}) but got indices tensor with shape: {indices.shape}')
    for i in range(1, input.ndim):
        torch._check(input.size(i) > 0, lambda: f'{fn_name}: Expected input to have non-zero size for non-batch dimensions, but got {input.shape} with dimension {i} being empty.')
    torch._check(stride[0] > 0 and stride[1] > 0 and (stride[2] > 0), lambda: f'strides should be greater than zero, but got stride: {stride}')