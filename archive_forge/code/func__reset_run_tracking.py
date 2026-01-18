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
def _reset_run_tracking(self):
    """Reset the counters and flags used to track the execution."""
    with getattr(self, '_lock', nullcontext()):
        if self._running:
            msg = 'This Parallel instance is already running !'
            if self.return_generator is True:
                msg += ' Before submitting new tasks, you must wait for the completion of all the previous tasks, or clean all references to the output generator.'
            raise RuntimeError(msg)
        self._running = True
    self.n_dispatched_batches = 0
    self.n_dispatched_tasks = 0
    self.n_completed_tasks = 0
    self._nb_consumed = 0
    self._exception = False
    self._aborting = False
    self._aborted = False