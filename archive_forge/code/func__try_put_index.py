import functools
import itertools
import logging
import os
import queue
import threading
import warnings
from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union
import multiprocessing as python_multiprocessing
import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
import torch.utils.data.graph_settings
from torch._utils import ExceptionWrapper
from . import (
from torch.utils.data.datapipes.datapipe import _IterDataPipeSerializationWrapper, _MapDataPipeSerializationWrapper
from . import _utils
def _try_put_index(self):
    assert self._tasks_outstanding < self._prefetch_factor * self._num_workers
    try:
        index = self._next_index()
    except StopIteration:
        return
    for _ in range(self._num_workers):
        worker_queue_idx = next(self._worker_queue_idx_cycle)
        if self._workers_status[worker_queue_idx]:
            break
    else:
        return
    self._index_queues[worker_queue_idx].put((self._send_idx, index))
    self._task_info[self._send_idx] = (worker_queue_idx,)
    self._tasks_outstanding += 1
    self._send_idx += 1