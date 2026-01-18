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
def _upsample_get_cubic_coefficients(t: Tensor) -> TensorSequenceType:
    A = -0.75
    return (_upsample_cubic_convolution2(t + 1.0, A), _upsample_cubic_convolution1(t, A), _upsample_cubic_convolution1(1.0 - t, A), _upsample_cubic_convolution2(2.0 - t, A))