import abc
import functools
import json
import os
import signal
import socket
import time
import traceback
import warnings
from contextlib import closing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch.distributed.elastic.rendezvous as rdzv
import torch.distributed.elastic.utils.store as store_util
from torch.distributed import Store
from torch.distributed.elastic.events import Event, EventSource, record
from torch.distributed.elastic.metrics import prof, put_metric
from torch.distributed.elastic.multiprocessing import (
from torch.distributed.elastic.utils.logging import get_logger
@prof
def _rendezvous(self, worker_group: WorkerGroup) -> None:
    """Run rendezvous for the workers specified by the worker spec.

        Assigns workers a new global rank and world size.
        Updates the rendezvous store for the worker group.
        """
    spec = worker_group.spec
    store, group_rank, group_world_size = spec.rdzv_handler.next_rendezvous()
    self._store = store
    workers = self._assign_worker_ranks(store, group_rank, group_world_size, spec)
    worker_group.workers = workers
    worker_group.store = store
    worker_group.group_rank = group_rank
    worker_group.group_world_size = group_world_size
    if group_rank == 0:
        self._set_master_addr_port(store, spec.master_addr, spec.master_port, spec.local_addr)
    master_addr, master_port = self._get_master_addr_port(store)
    restart_count = spec.max_restarts - self._remaining_restarts
    log.info('[%(role)s] Rendezvous complete for workers. Result:\n  restart_count=%(restart_count)s\n  master_addr=%(master_addr)s\n  master_port=%(master_port)s\n  group_rank=%(group_rank)s\n  group_world_size=%(group_world_size)s\n  local_ranks=%(local_ranks)s\n  role_ranks=%(role_ranks)s\n  global_ranks=%(global_ranks)s\n  role_world_sizes=%(role_world_sizes)s\n  global_world_sizes=%(global_world_sizes)s\n', {'role': spec.role, 'restart_count': restart_count, 'master_addr': master_addr, 'master_port': master_port, 'group_rank': group_rank, 'group_world_size': group_world_size, 'local_ranks': [worker.local_rank for worker in workers], 'role_ranks': [worker.role_rank for worker in workers], 'global_ranks': [worker.global_rank for worker in workers], 'role_world_sizes': [worker.role_world_size for worker in workers], 'global_world_sizes': [worker.world_size for worker in workers]})