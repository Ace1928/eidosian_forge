import math
import warnings
from typing import Any, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from peft.tuners.lycoris_utils import LycorisLayer, check_adapters_to_merge
def _cayley_batch(self, data: torch.Tensor) -> torch.Tensor:
    b, r, c = data.shape
    skew = 0.5 * (data - data.transpose(1, 2))
    I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)
    Q = torch.bmm(I - skew, torch.inverse(I + skew))
    return Q