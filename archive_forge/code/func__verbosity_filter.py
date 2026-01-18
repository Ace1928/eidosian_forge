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
def _verbosity_filter(index, verbose):
    """ Returns False for indices increasingly apart, the distance
        depending on the value of verbose.

        We use a lag increasing as the square of index
    """
    if not verbose:
        return True
    elif verbose > 10:
        return False
    if index == 0:
        return False
    verbose = 0.5 * (11 - verbose) ** 2
    scale = sqrt(index / verbose)
    next_scale = sqrt((index + 1) / verbose)
    return int(next_scale) == int(scale)