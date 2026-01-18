import math
import os
import re
import dis
import numbers
import platform
import sys
import subprocess
import types as pytypes
import warnings
from functools import reduce
import numpy as np
from numpy.random import randn
import operator
from collections import defaultdict, namedtuple
import copy
from itertools import cycle, chain
import subprocess as subp
import numba.parfors.parfor
from numba import (njit, prange, parallel_chunksize,
from numba.core import (types, errors, ir, rewrites,
from numba.extending import (overload_method, register_model,
from numba.core.registry import cpu_target
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (find_callname, guard, build_definitions,
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty
from numba.core.compiler import (CompilerBase, DefaultPassBuilder)
from numba.core.compiler_machinery import register_pass, AnalysisPass
from numba.core.typed_passes import IRLegalization
from numba.tests.support import (TestCase, captured_stdout, MemoryLeakMixin,
from numba.core.extending import register_jitable
from numba.core.bytecode import _fix_LOAD_GLOBAL_arg
from numba.core import utils
import cmath
import unittest
@skip_parfors_unsupported
class TestParforsLeaks(MemoryLeakMixin, TestParforsBase):

    def check(self, pyfunc, *args, **kwargs):
        cfunc, cpfunc = self.compile_all(pyfunc, *args)
        self.check_parfors_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)

    def test_reduction(self):

        def test_impl(arr):
            return arr.sum()
        arr = np.arange(10).astype(np.float64)
        self.check(test_impl, arr)

    def test_multiple_reduction_vars(self):

        def test_impl(arr):
            a = 0.0
            b = 1.0
            for i in prange(arr.size):
                a += arr[i]
                b += 1.0 / (arr[i] + 1)
            return a * b
        arr = np.arange(10).astype(np.float64)
        self.check(test_impl, arr)