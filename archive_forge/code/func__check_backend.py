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
def _check_backend(self, backend, inner_max_num_threads, **backend_params):
    if backend is default_parallel_config['backend']:
        if inner_max_num_threads is not None or len(backend_params) > 0:
            raise ValueError('inner_max_num_threads and other constructor parameters backend_params are only supported when backend is not None.')
        return backend
    if isinstance(backend, str):
        if backend not in BACKENDS:
            if backend in EXTERNAL_BACKENDS:
                register = EXTERNAL_BACKENDS[backend]
                register()
            elif backend in MAYBE_AVAILABLE_BACKENDS:
                warnings.warn(f"joblib backend '{backend}' is not available on your system, falling back to {DEFAULT_BACKEND}.", UserWarning, stacklevel=2)
                BACKENDS[backend] = BACKENDS[DEFAULT_BACKEND]
            else:
                raise ValueError(f'Invalid backend: {backend}, expected one of {sorted(BACKENDS.keys())}')
        backend = BACKENDS[backend](**backend_params)
    if inner_max_num_threads is not None:
        msg = f'{backend.__class__.__name__} does not accept setting the inner_max_num_threads argument.'
        assert backend.supports_inner_max_num_threads, msg
        backend.inner_max_num_threads = inner_max_num_threads
    if backend.nesting_level is None:
        parent_backend = self.old_parallel_config['backend']
        if parent_backend is default_parallel_config['backend']:
            nesting_level = 0
        else:
            nesting_level = parent_backend.nesting_level
        backend.nesting_level = nesting_level
    return backend