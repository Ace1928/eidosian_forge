from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from .common import BaseOperator, get_xformers_operator, register_operator
from .unbind import stack_or_none, unbind
@register_operator
class DualGemmSiluOp(BaseOperator):
    OPERATOR = get_xformers_operator('dual_gemm_silu_identity_mul')
    OPERATOR_CATEGORY = 'swiglu'
    NAME = 'dual_gemm_silu'

    @classmethod
    def operator_flop(cls, x: torch.Tensor, w1: torch.Tensor, b1, w2: torch.Tensor, b2) -> int:
        """NOTE: we neglect the impact of biases / pointwises"""
        M, N, K = (x.shape[0], w1.shape[0], w1.shape[1])
        return M * N * K * 2 * 2