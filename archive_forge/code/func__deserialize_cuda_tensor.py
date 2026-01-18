import concurrent.futures
import json
import multiprocessing.connection
from typing import Any, List, Optional, Union
import torch
import torch.distributed as dist
import torch.multiprocessing.reductions
def _deserialize_cuda_tensor(args, device: torch.device) -> torch.Tensor:
    args = list(args)
    args[6] = device.index
    return torch.multiprocessing.reductions.rebuild_cuda_tensor(*args)