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
@skip_if_typeguard
class TestOverloadMethodCaching(TestCase):
    _numba_parallel_test_ = False

    def test_caching_overload_method(self):
        self._cache_dir = temp_directory(self.__class__.__name__)
        with override_config('CACHE_DIR', self._cache_dir):
            self.run_caching_overload_method()

    def run_caching_overload_method(self):
        cfunc = jit(nopython=True, cache=True)(cache_overload_method_usecase)
        self.assertPreciseEqual(cfunc(MyDummy()), 13)
        _assert_cache_stats(cfunc, 0, 1)
        llvmir = cfunc.inspect_llvm((mydummy_type,))
        decls = [ln for ln in llvmir.splitlines() if ln.startswith('declare') and 'overload_method_length' in ln]
        self.assertEqual(len(decls), 0)
        try:
            ctx = multiprocessing.get_context('spawn')
        except AttributeError:
            ctx = multiprocessing
        q = ctx.Queue()
        p = ctx.Process(target=run_caching_overload_method, args=(q, self._cache_dir))
        p.start()
        q.put(MyDummy())
        p.join()
        self.assertEqual(p.exitcode, 0)
        res = q.get(timeout=1)
        self.assertEqual(res, 13)