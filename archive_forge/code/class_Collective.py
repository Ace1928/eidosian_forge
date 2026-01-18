import sys
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from functools import partial, reduce
import torch
import torch.distributed as dist
import weakref
from torch._C._distributed_c10d import (
from torch.distributed.distributed_c10d import _CollOp, _store_based_barrier, P2POp
from torch.futures import Future
from torch.utils import _pytree as pytree
class Collective:

    def __init__(self, world_size, collective, pg):
        self._world_size = world_size
        self._collective = collective
        self._start_cond = threading.Condition()
        self._done_cond = threading.Condition()
        self._data = [None] * world_size
        self._count = 0
        self._done = False
        self._pg = pg

    def join(self, rank, data):
        with self._start_cond:
            self._data[rank] = data
            self._count += 1
            if self._count == self._world_size:
                if rank > 0:
                    self._start_cond.notify()
            if rank == 0:
                self._start_cond.wait_for(lambda: self._count == self._world_size or self._pg._terminate.is_set())
                if self._pg._terminate.is_set():
                    sys.exit('Test termination event occurs.')
        with self._done_cond:
            if rank > 0:
                self._done_cond.wait_for(lambda: self._done or self._pg._terminate.is_set())
                if self._pg._terminate.is_set():
                    sys.exit('Test termination event occurs.')
            else:
                self._collective.work(self._data)
                self._done = True
                self._done_cond.notify_all()
        return ret_work(data)