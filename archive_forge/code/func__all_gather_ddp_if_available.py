import contextlib
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, List, Optional, Sized, Union
import torch
import torch.nn.functional as F
from lightning_utilities.core.imports import package_available
from torch import Tensor
from torch.utils.data import Dataset, DistributedSampler, Sampler
from typing_extensions import override
from lightning_fabric.utilities.cloud_io import _is_local_file_protocol
from lightning_fabric.utilities.data import _num_cpus_available
from lightning_fabric.utilities.rank_zero import rank_zero_info
from lightning_fabric.utilities.types import _PATH, ReduceOp
def _all_gather_ddp_if_available(tensor: Tensor, group: Optional['torch.distributed.ProcessGroup']=None, sync_grads: bool=False) -> Tensor:
    """Function to gather a tensor from several distributed processes.

    Args:
        tensor: Tensor of shape (batch, ...)
        group: The process group to gather results from. Defaults to all processes (world)
        sync_grads: Flag that allows users to synchronize gradients for all_gather op

    Return:
        A tensor of shape (world_size, batch, ...)

    """
    if not _distributed_is_initialized():
        return tensor
    from torch.distributed.nn.functional import all_gather
    tensor = tensor.contiguous()
    with nullcontext() if sync_grads else torch.no_grad():
        gathered_tensors = all_gather(tensor, group)
    return torch.stack(gathered_tensors)