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
@register_decomposition(aten._foreach_addcdiv.Scalar)
def _foreach_addcdiv_scalar(self, left_tensors, right_tensors, scalar=1):
    return aten._foreach_add.List(self, aten._foreach_div.List(left_tensors, right_tensors), alpha=scalar)