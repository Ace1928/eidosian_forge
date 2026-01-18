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
def _set_nccl_network_interface():
    """Set the appropriate NCCL network interface to use."""
    if 'NCCL_SOCKET_IFNAME' not in os.environ:
        logger.debug(f"Setting NCCL_SOCKET_IFNAME to {DEFAULT_NCCL_SOCKET_IFNAME} to prioritize ethernet connection. To override this behavior, set the `NCCL_SOCKET_IFNAME` environment variable in your Ray runtime environment: `ray.init(runtime_env={{{{'env_vars': {{'NCCL_SOCKET_IFNAME': 'ens5'}}}}}}`")
        os.environ['NCCL_SOCKET_IFNAME'] = DEFAULT_NCCL_SOCKET_IFNAME