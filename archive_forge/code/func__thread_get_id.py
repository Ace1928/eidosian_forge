from __future__ import annotations
import atexit
import multiprocessing.pool
import sys
import threading
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from threading import Lock, current_thread
from dask import config
from dask.local import MultiprocessingPoolExecutor, get_async
from dask.system import CPU_COUNT
from dask.typing import Key
def _thread_get_id():
    return current_thread().ident