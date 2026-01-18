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
def _wait_retrieval(self):
    """Return True if we need to continue retriving some tasks."""
    if self._iterating:
        return True
    if self.n_completed_tasks < self.n_dispatched_tasks:
        return True
    if not self._backend.supports_retrieve_callback:
        if len(self._jobs) > 0:
            return True
    return False