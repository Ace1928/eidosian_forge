import logging
import warnings
import weakref
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import List, Optional, cast
def _wait_tensor(tensor: torch.Tensor) -> torch.Tensor:
    global data_ptr_to_work
    data_ptr = tensor.data_ptr()
    wait_reg = data_ptr_to_work.get(data_ptr)
    if wait_reg is not None:
        wait_reg.wait()
    return tensor