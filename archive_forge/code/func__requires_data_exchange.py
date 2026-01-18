from typing import cast, Dict, List, Tuple
import torch
import torch.distributed as dist
import torch.distributed._tensor.api as dtensor
def _requires_data_exchange(padding):
    return padding[1] != 0