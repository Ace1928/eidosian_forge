import math
from functools import wraps
from typing import Callable, Optional, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs
from torch._decomp import register_decomposition
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _make_inplace
def _dropout_helper(self: TensorLikeType, val: float) -> TensorLikeType:
    """
    Helper function for all dropout-type operators. During training,
    some of the elements of the input tensor are randomly masked.

    Returns the masked tensor of the boolean values.

    """
    return refs._uniform_helper(self.shape, low=0.0, high=1.0, dtype=torch.float32, device=self.device) < val