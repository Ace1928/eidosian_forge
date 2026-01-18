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
def _dispatch_new(self):
    """Schedule the next batch of tasks to be processed."""
    this_batch_duration = time.time() - self.dispatch_timestamp
    self.parallel._backend.batch_completed(self.batch_size, this_batch_duration)
    with self.parallel._lock:
        self.parallel.n_completed_tasks += self.batch_size
        self.parallel.print_progress()
        if self.parallel._original_iterator is not None:
            self.parallel.dispatch_next()