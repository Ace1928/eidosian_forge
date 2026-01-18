import functools
import logging
import math
import sys
import typing
from typing import Optional
import torch
import torch._decomp as decomp
import torch._prims_common as utils
import torch.ao.quantization.fx._decomposed
from torch._decomp import (
from torch._decomp.decompositions import (
from torch._decomp.decompositions_for_rng import extra_random_decomps
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import type_to_dtype
from . import config, inductor_prims
@register_decomposition(aten._foreach_lerp.Scalar)
def _foreach_lerp_scalar(start_tensors, end_tensors, weight):
    return aten._foreach_add.List(start_tensors, aten._foreach_mul.Scalar(aten._foreach_sub.List(end_tensors, start_tensors), weight))