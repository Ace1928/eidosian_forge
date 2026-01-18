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
def _get_active_backend(prefer=default_parallel_config['prefer'], require=default_parallel_config['require'], verbose=default_parallel_config['verbose']):
    """Return the active default backend"""
    backend_config = getattr(_backend, 'config', default_parallel_config)
    backend = _get_config_param(default_parallel_config['backend'], backend_config, 'backend')
    prefer = _get_config_param(prefer, backend_config, 'prefer')
    require = _get_config_param(require, backend_config, 'require')
    verbose = _get_config_param(verbose, backend_config, 'verbose')
    if prefer not in VALID_BACKEND_HINTS:
        raise ValueError(f'prefer={prefer} is not a valid backend hint, expected one of {VALID_BACKEND_HINTS}')
    if require not in VALID_BACKEND_CONSTRAINTS:
        raise ValueError(f'require={require} is not a valid backend constraint, expected one of {VALID_BACKEND_CONSTRAINTS}')
    if prefer == 'processes' and require == 'sharedmem':
        raise ValueError("prefer == 'processes' and require == 'sharedmem' are inconsistent settings")
    explicit_backend = True
    if backend is None:
        backend = BACKENDS[DEFAULT_BACKEND](nesting_level=0)
        explicit_backend = False
    nesting_level = backend.nesting_level
    uses_threads = getattr(backend, 'uses_threads', False)
    supports_sharedmem = getattr(backend, 'supports_sharedmem', False)
    force_threads = require == 'sharedmem' and (not supports_sharedmem)
    force_threads |= not explicit_backend and prefer == 'threads' and (not uses_threads)
    if force_threads:
        sharedmem_backend = BACKENDS[DEFAULT_THREAD_BACKEND](nesting_level=nesting_level)
        if verbose >= 10 and explicit_backend:
            print(f'Using {sharedmem_backend.__class__.__name__} as joblib backend instead of {backend.__class__.__name__} as the latter does not provide shared memory semantics.')
        thread_config = backend_config.copy()
        thread_config['n_jobs'] = 1
        return (sharedmem_backend, thread_config)
    return (backend, backend_config)