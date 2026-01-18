from contextlib import contextmanager
from datetime import timedelta
from functools import (
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

    Function wrapper that inits a fake process group designed for testing.
    Right now only querying for world size is available
    