from contextlib import contextmanager
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import distributed_c10d
from torch.distributed._shard.sharded_tensor import (
from .sharding_spec import (
from .sharding_plan import (
from .sharder import Sharder
def _get_current_process_group():
    """
    Retrieves the current process group set by ``load_with_process_group``.
    If not set, it just returns the default group.
    """
    global _CURRENT_PROCESS_GROUP
    if _CURRENT_PROCESS_GROUP is None:
        return distributed_c10d._get_default_group()
    else:
        return _CURRENT_PROCESS_GROUP