from __future__ import division
import os
import sys
from math import sqrt
import functools
import collections
import time
import threading
import itertools
from uuid import uuid4
from numbers import Integral
import warnings
import queue
import weakref
from contextlib import nullcontext
from multiprocessing import TimeoutError
from ._multiprocessing_helpers import mp
from .logger import Logger, short_format_time
from .disk import memstr_to_bytes
from ._parallel_backends import (FallbackToBackend, MultiprocessingBackend,
from ._utils import eval_expr, _Sentinel
from ._parallel_backends import AutoBatchingMixin  # noqa
from ._parallel_backends import ParallelBackendBase  # noqa
def _register_dask():
    """Register Dask Backend if called with parallel_config(backend="dask")"""
    try:
        from ._dask import DaskDistributedBackend
        register_parallel_backend('dask', DaskDistributedBackend)
    except ImportError as e:
        msg = 'To use the dask.distributed backend you must install both the `dask` and distributed modules.\n\nSee https://dask.pydata.org/en/latest/install.html for more information.'
        raise ImportError(msg) from e