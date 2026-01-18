from contextlib import contextmanager
from datetime import timedelta
from functools import (
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
class MockProcessGroup(dist.ProcessGroup):

    def __init__(self, rank, world):
        super().__init__(rank, world)

    def getBackendName(self):
        return 'mock_process_group'