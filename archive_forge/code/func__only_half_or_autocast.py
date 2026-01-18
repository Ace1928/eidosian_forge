from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from .common import BaseOperator, get_xformers_operator, register_operator
from .unbind import stack_or_none, unbind
def _only_half_or_autocast(op: SwiGLUOpDispatch) -> bool:
    HALF_DTYPES = [torch.half, torch.bfloat16]
    return op.dtype in HALF_DTYPES or (op.dtype_autocast_gpu is not None and op.dtype_autocast_gpu in HALF_DTYPES)