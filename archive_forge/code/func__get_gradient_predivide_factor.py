import functools
import torch
import torch.distributed as dist
from typing import Optional
@staticmethod
def _get_gradient_predivide_factor(world_size: int) -> float:
    factor: int = 1
    while world_size % factor == 0 and world_size / factor > factor:
        factor *= 2
    return float(factor)