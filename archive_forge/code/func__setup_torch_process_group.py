import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional
import torch
import torch.distributed as dist
import ray
from ray.train._internal.utils import get_address_and_port
from ray.train._internal.worker_group import WorkerGroup
from ray.train.backend import Backend, BackendConfig
from ray.train.constants import DEFAULT_NCCL_SOCKET_IFNAME
from ray.util import PublicAPI
def _setup_torch_process_group(backend: str, world_rank: int, world_size: int, init_method: str, timeout_s: int=1800):
    """Connects the distributed PyTorch backend.

    Args:
        backend: The backend (nccl, gloo, etc.) to use for training.
        world_rank: Rank of the current worker.
        world_size: Number of workers participating in the job.
        init_method: URL specifying how to initialize the process group.
        timeout_s: Seconds for process group operations to timeout.
    """
    if world_rank == 0:
        logger.info(f'Setting up process group for: {init_method} [rank={world_rank}, world_size={world_size}]')
    else:
        logger.debug(f'Setting up process group for: {init_method} [rank={world_rank}, world_size={world_size}]')
    logger.debug(f'using {backend}')
    if backend == 'nccl' and 'NCCL_ASYNC_ERROR_HANDLING' not in os.environ and ('NCCL_BLOCKING_WAIT' not in os.environ):
        logger.debug('Setting NCCL_ASYNC_ERROR_HANDLING to fail if NCCL collective communication operations are timing out. To override this behavior, you can set NCCL_ASYNC_ERROR_HANDLING=0.')
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    dist.init_process_group(backend=backend, init_method=init_method, rank=world_rank, world_size=world_size, timeout=timedelta(seconds=timeout_s))