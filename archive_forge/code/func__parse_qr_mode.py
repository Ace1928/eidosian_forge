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
def _parse_qr_mode(mode: str) -> Tuple[bool, bool]:
    if mode == 'reduced':
        compute_q = True
        reduced = True
    elif mode == 'complete':
        compute_q = True
        reduced = False
    elif mode == 'r':
        compute_q = False
        reduced = True
    else:
        torch._check(False, lambda: f"qr received unrecognized mode '{mode}' but expected one of 'reduced' (default), 'r', or 'complete'")
    return (compute_q, reduced)