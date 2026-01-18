import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from vllm._C import ops
def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)