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
def _warn_exit_early(self):
    """Warn the user if the generator is gc'ed before being consumned."""
    ready_outputs = self.n_completed_tasks - self._nb_consumed
    is_completed = self._is_completed()
    msg = ''
    if ready_outputs:
        msg += f'{ready_outputs} tasks have been successfully executed  but not used.'
        if not is_completed:
            msg += ' Additionally, '
    if not is_completed:
        msg += f'{self.n_dispatched_tasks - self.n_completed_tasks} tasks which were still being processed by the workers have been cancelled.'
    if msg:
        msg += ' You could benefit from adjusting the input task iterator to limit unnecessary computation time.'
        warnings.warn(msg)