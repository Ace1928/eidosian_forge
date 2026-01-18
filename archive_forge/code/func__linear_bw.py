from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from .common import BaseOperator, get_xformers_operator, register_operator
from .unbind import stack_or_none, unbind
@staticmethod
def _linear_bw(dy: torch.Tensor, x: torch.Tensor, bias: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not bias:
        return (dy.transpose(-2, -1) @ x, None)
    db = torch.empty([dy.shape[1]], dtype=dy.dtype, device=dy.device)
    dw = torch.empty([dy.shape[1], x.shape[1]], dtype=dy.dtype, device=dy.device)
    GemmFusedSumOp.OPERATOR(dy.transpose(-2, -1), x, dw, db)
    return (dw, db)