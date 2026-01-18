from __future__ import annotations
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor, Future
from functools import partial
from queue import Empty, Queue
from dask import config
from dask.callbacks import local_callbacks, unpack_callbacks
from dask.core import _execute_task, flatten, get_dependencies, has_tasks, reverse_dict
from dask.order import order
from dask.typing import Key
class SynchronousExecutor(Executor):
    _max_workers = 1

    def submit(self, fn, *args, **kwargs):
        fut = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as e:
            fut.set_exception(e)
        return fut