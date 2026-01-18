import inspect
import math
import operator
import sys
import pickle
import multiprocessing
import ctypes
import warnings
import re
import numpy as np
from llvmlite import ir
import numba
from numba import njit, jit, vectorize, guvectorize, objmode
from numba.core import types, errors, typing, compiler, cgutils
from numba.core.typed_passes import type_inference_stage
from numba.core.registry import cpu_target
from numba.core.imputils import lower_constant
from numba.tests.support import (
from numba.core.errors import LoweringError
import unittest
from numba.extending import (
from numba.core.typing.templates import (
from .pdlike_usecase import Index, Series
def _assert_cache_stats(cfunc, expect_hit, expect_misses):
    hit = cfunc._cache_hits[cfunc.signatures[0]]
    if hit != expect_hit:
        raise AssertionError('cache not used')
    miss = cfunc._cache_misses[cfunc.signatures[0]]
    if miss != expect_misses:
        raise AssertionError('cache not used')