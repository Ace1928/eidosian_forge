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
@PublicAPI(stability='stable')
@dataclass
class TorchConfig(BackendConfig):
    """Configuration for torch process group setup.

    See https://pytorch.org/docs/stable/distributed.html for more info.

    Args:
        backend: The backend to use for training.
            See ``torch.distributed.init_process_group`` for more info and
            valid values.
            If set to None, nccl will be used if GPUs are requested, else gloo
            will be used.
        init_method: The initialization method to use. Either "env"
            for environment variable initialization or "tcp" for TCP
            initialization. Defaults to "env".
        timeout_s: Seconds for process group operations to timeout.
    """
    backend: Optional[str] = None
    init_method: str = 'env'
    timeout_s: int = 1800

    @property
    def backend_cls(self):
        return _TorchBackend