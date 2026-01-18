from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from .common import BaseOperator, get_xformers_operator, register_operator
from .unbind import stack_or_none, unbind
def _ordered_params(self) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    """Used for testing - returns ordered arguments for operators"""
    b1: Optional[torch.Tensor]
    b2: Optional[torch.Tensor]
    if self.w12 is not None:
        w1w2 = self.w12.weight
        b1b2 = self.w12.bias
        w1, w2 = unbind(w1w2.view([2, w1w2.shape[0] // 2, w1w2.shape[1]]), dim=0)
        if b1b2 is not None:
            b1, b2 = unbind(b1b2.view([2, b1b2.shape[0] // 2]), dim=0)
        else:
            b1, b2 = (None, None)
    else:
        w1, w2 = (self.w1.weight, self.w2.weight)
        b1, b2 = (self.w1.bias, self.w2.bias)
    return (w1, b1, w2, b2, self.w3.weight, self.w3.bias)