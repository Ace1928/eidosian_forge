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
def _sharding_worker_init_fn(worker_init_fn, world_size, rank_id, worker_id):
    global_worker_id = worker_id
    info = torch.utils.data.get_worker_info()
    assert info is not None
    total_workers = info.num_workers
    datapipe = info.dataset
    assert isinstance(datapipe, (IterDataPipe, MapDataPipe))
    total_workers *= world_size
    global_worker_id = global_worker_id * world_size + rank_id
    torch.utils.data.graph_settings.apply_sharding(datapipe, total_workers, global_worker_id)
    if worker_init_fn is not None:
        worker_init_fn(worker_id)