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
@register_decomposition(aten._prelu_kernel_backward)
def _prelu_kernel_backward(grad_output: Tensor, self: Tensor, weight: Tensor) -> Tuple[Tensor, Tensor]:
    input_grad = torch.where(self > 0, grad_output, weight * grad_output)
    weight_grad = torch.where(self > 0, 0.0, self * grad_output)
    return (input_grad, weight_grad)