from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from .common import BaseOperator, get_xformers_operator, register_operator
from .unbind import stack_or_none, unbind
class SwiGLUOp:
    """Base class for any swiglu operator in :attr:`xformers.ops.swiglu`"""

    def __init__(self, op, packed_weights: bool, name: str, constraints):
        self.NAME = name
        self.PACKED_WEIGHTS = packed_weights
        self.op = op
        self.constraints = constraints

    def supports(self, op: 'SwiGLUOpDispatch') -> bool:
        if self.PACKED_WEIGHTS and (not op.packed_weights):
            return False
        return all((c(op) for c in self.constraints))

    def __call__(self, *args: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f'SwiGLUOp:{self.NAME}'