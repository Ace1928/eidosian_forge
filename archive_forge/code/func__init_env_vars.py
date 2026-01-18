import os
from dataclasses import dataclass
from typing import Optional, Set
from horovod.ray.runner import Coordinator
from horovod.ray.utils import detect_nics, nics_to_env_var
from horovod.runner.common.util import secret, timeout
import ray
from ray.train._internal.utils import update_env_vars
from ray.train._internal.worker_group import Worker, WorkerGroup
from ray.train.backend import Backend, BackendConfig
from ray.util import PublicAPI
def _init_env_vars(world_rank: int, world_size: int, node_id: str):
    """Initialize Horovod environment variables."""
    os.environ['HOROVOD_HOSTNAME'] = node_id
    os.environ['HOROVOD_RANK'] = str(world_rank)
    os.environ['HOROVOD_SIZE'] = str(world_size)