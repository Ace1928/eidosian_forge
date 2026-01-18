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
def _get_config_param(param, context_config, key):
    """Return the value of a parallel config parameter

    Explicitly setting it in Parallel has priority over setting in a
    parallel_(config/backend) context manager.
    """
    if param is not default_parallel_config[key]:
        return param
    if context_config[key] is not default_parallel_config[key]:
        return context_config[key]
    return param.default_value