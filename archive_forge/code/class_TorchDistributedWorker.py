from abc import ABC
from collections import defaultdict
from datetime import timedelta
import os
import torch
import torch.distributed as dist
from typing import Callable, List, T
import ray
from ray.actor import ActorHandle
from ray.train._internal.utils import get_address_and_port
from ray.train.constants import DEFAULT_NCCL_SOCKET_IFNAME
from ray.air._internal.torch_utils import get_device
class TorchDistributedWorker(ABC):
    """Defines the interfaces required by the init_torch_dist_process_group().

    This is modeled after RayTrainerWorker, which allows arbitrary functions
    to be executed on a remote DDP worker.
    """

    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Executes the input function and returns the output.

        Args:
            func: The function to execute.
            args, kwargs: The arguments to pass into func.
        """
        return func(*args, **kwargs)