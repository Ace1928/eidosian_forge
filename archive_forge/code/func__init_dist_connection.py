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
def _init_dist_connection(cluster_environment: 'ClusterEnvironment', torch_distributed_backend: str, global_rank: Optional[int]=None, world_size: Optional[int]=None, **kwargs: Any) -> None:
    """Utility function to initialize distributed connection by setting env variables and initializing the distributed
    process group.

    Args:
        cluster_environment: ``ClusterEnvironment`` instance
        torch_distributed_backend: Backend to use (includes `nccl` and `gloo`)
        global_rank: Rank of the current process
        world_size: Number of processes in the group
        kwargs: Kwargs for ``init_process_group``

    Raises:
        RuntimeError:
            If ``torch.distributed`` is not available

    """
    if not torch.distributed.is_available():
        raise RuntimeError('torch.distributed is not available. Cannot initialize distributed process group')
    if torch.distributed.is_initialized():
        log.debug('torch.distributed is already initialized. Exiting early')
        return
    global_rank = global_rank if global_rank is not None else cluster_environment.global_rank()
    world_size = world_size if world_size is not None else cluster_environment.world_size()
    os.environ['MASTER_ADDR'] = cluster_environment.main_address
    os.environ['MASTER_PORT'] = str(cluster_environment.main_port)
    log.info(f'Initializing distributed: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}')
    torch.distributed.init_process_group(torch_distributed_backend, rank=global_rank, world_size=world_size, **kwargs)
    rank_zero_info(f'{'-' * 100}\ndistributed_backend={torch_distributed_backend}\nAll distributed processes registered. Starting with {world_size} processes\n{'-' * 100}\n')