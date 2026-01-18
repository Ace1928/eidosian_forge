from typing import List, Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn
from . import blockpartition
from .profile import profile_sizes, profile_times
def _balance_cost(cost: List[int], partitions: int) -> List[int]:
    partitioned = blockpartition.solve(cost, partitions)
    return [len(p) for p in partitioned]