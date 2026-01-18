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
class TestParforsBitMask(TestParforsBase):

    def test_parfor_bitmask1(self):

        def test_impl(a, n):
            b = a > n
            a[b] = 0
            return a
        self.check(test_impl, np.arange(10), 5)

    def test_parfor_bitmask2(self):

        def test_impl(a, b):
            a[b] = 0
            return a
        a = np.arange(10)
        b = a > 5
        self.check(test_impl, a, b)

    def test_parfor_bitmask3(self):

        def test_impl(a, b):
            a[b] = a[b]
            return a
        a = np.arange(10)
        b = a > 5
        self.check(test_impl, a, b)

    def test_parfor_bitmask4(self):

        def test_impl(a, b):
            a[b] = (2 * a)[b]
            return a
        a = np.arange(10)
        b = a > 5
        self.check(test_impl, a, b)

    def test_parfor_bitmask5(self):

        def test_impl(a, b):
            a[b] = a[b] * a[b]
            return a
        a = np.arange(10)
        b = a > 5
        self.check(test_impl, a, b)

    def test_parfor_bitmask6(self):

        def test_impl(a, b, c):
            a[b] = c
            return a
        a = np.arange(10)
        b = a > 5
        c = np.zeros(sum(b))
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl, a, b, c)
        self.assertIn("'@do_scheduling' not found", str(raises.exception))