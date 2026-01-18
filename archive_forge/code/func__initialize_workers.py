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
def _initialize_workers(self, worker_group: WorkerGroup) -> None:
    """Start a fresh set of workers for the worker_group.

        Essentially, a rendezvous followed by a ``start_workers``.
        The caller should first call ``_stop_workers()`` to stop running workers
        prior to calling this method.

        Optimistically sets the state of the worker group that
        just started as ``HEALTHY`` and delegates the actual monitoring
        of state to ``_monitor_workers()`` method
        """
    role = worker_group.spec.role
    log.info("[%s] Rendezvous'ing worker group", role)
    self._rendezvous(worker_group)
    log.info('[%s] Starting worker group', role)
    worker_ids = self._start_workers(worker_group)
    for local_rank, w_id in worker_ids.items():
        worker = worker_group.workers[local_rank]
        worker.id = w_id
    worker_group.state = WorkerState.HEALTHY