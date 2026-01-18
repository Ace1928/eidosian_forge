from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
def _upcast_non_float(t: Tensor) -> Tensor:
    if t.dtype not in (torch.float32, torch.float64):
        return t.float()
    return t