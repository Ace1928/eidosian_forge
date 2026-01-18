from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from .common import BaseOperator, get_xformers_operator, register_operator
from .unbind import stack_or_none, unbind
@dataclass
class SwiGLUOpDispatch:
    """Dispatcher to automatically select
    the best operator in :attr:`xformers.ops.swiglu`
    """
    device: Union[torch.device, str]
    dtype: torch.dtype
    dtype_autocast_gpu: Optional[torch.dtype]
    packed_weights: bool
    bias_enabled: bool

    @property
    def op(self) -> SwiGLUOp:
        """Computes the best operator

        Returns:
            SwiGLUOp: The best operator for the configuration
        """
        priorities: Sequence[SwiGLUOp] = [SwiGLUPackedFusedOp, SwiGLUFusedOp]
        for op in priorities:
            if op.supports(self):
                return op
        return SwiGLUEagerOp

    @staticmethod
    def from_arguments(x: torch.Tensor, w1: torch.Tensor, b1: Optional[torch.Tensor], w2: torch.Tensor, b2: Optional[torch.Tensor], w3: torch.Tensor, b3: Optional[torch.Tensor]) -> 'SwiGLUOpDispatch':
        return SwiGLUOpDispatch(device=x.device, dtype=x.dtype, packed_weights=stack_or_none((w1, w2), dim=0) is not None, dtype_autocast_gpu=torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else w1.dtype, bias_enabled=b1 is not None and b2 is not None and (b3 is not None))