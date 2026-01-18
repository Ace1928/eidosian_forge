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
def _sync_ddp_if_available(result: Tensor, group: Optional[Any]=None, reduce_op: Optional[Union[ReduceOp, str]]=None) -> Tensor:
    """Function to reduce a tensor across worker processes during distributed training.

    Args:
        result: The value to sync and reduce (typically tensor or number)
        group: The process group to gather results from. Defaults to all processes (world)
        reduce_op: The reduction operation. Defaults to sum.
            Can also be a string of 'avg', 'mean' to calculate the mean during reduction.

    Return:
        reduced value

    """
    if _distributed_is_initialized():
        return _sync_ddp(result, group=group, reduce_op=reduce_op)
    return result