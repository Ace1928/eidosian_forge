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
class TestPrangeBase(TestParforsBase):

    def generate_prange_func(self, pyfunc, patch_instance):
        """
        This function does the actual code augmentation to enable the explicit
        testing of `prange` calls in place of `range`.
        """
        pyfunc_code = pyfunc.__code__
        prange_names = list(pyfunc_code.co_names)
        if patch_instance is None:
            assert 'range' in pyfunc_code.co_names
            prange_names = tuple([x if x != 'range' else 'prange' for x in pyfunc_code.co_names])
            new_code = bytes(pyfunc_code.co_code)
        else:
            range_idx = pyfunc_code.co_names.index('range')
            range_locations = []
            for instr in dis.Bytecode(pyfunc_code):
                if instr.opname == 'LOAD_GLOBAL':
                    if _fix_LOAD_GLOBAL_arg(instr.arg) == range_idx:
                        range_locations.append(instr.offset + 1)
            prange_names.append('prange')
            prange_names = tuple(prange_names)
            prange_idx = len(prange_names) - 1
            if utils.PYVERSION in ((3, 11), (3, 12)):
                prange_idx = 1 + (prange_idx << 1)
            elif utils.PYVERSION in ((3, 9), (3, 10)):
                pass
            else:
                raise NotImplementedError(utils.PYVERSION)
            new_code = bytearray(pyfunc_code.co_code)
            assert len(patch_instance) <= len(range_locations)
            for i in patch_instance:
                idx = range_locations[i]
                new_code[idx] = prange_idx
            new_code = bytes(new_code)
        prange_code = pyfunc_code.replace(co_code=new_code, co_names=prange_names)
        pfunc = pytypes.FunctionType(prange_code, globals())
        return pfunc

    def prange_tester(self, pyfunc, *args, **kwargs):
        """
        The `prange` tester
        This is a hack. It basically switches out range calls for prange.
        It does this by copying the live code object of a function
        containing 'range' then copying the .co_names and mutating it so
        that 'range' is replaced with 'prange'. It then creates a new code
        object containing the mutation and instantiates a function to contain
        it. At this point three results are created:
        1. The result of calling the original python function.
        2. The result of calling a njit compiled version of the original
            python function.
        3. The result of calling a njit(parallel=True) version of the mutated
           function containing `prange`.
        The three results are then compared and the `prange` based function's
        llvm_ir is inspected to ensure the scheduler code is present.

        Arguments:
         pyfunc - the python function to test
         args - data arguments to pass to the pyfunc under test

        Keyword Arguments:
         patch_instance - iterable containing which instances of `range` to
                          replace. If not present all instance of `range` are
                          replaced.
         scheduler_type - 'signed', 'unsigned' or None, default is None.
                           Supply in cases where the presence of a specific
                           scheduler is to be asserted.
         check_fastmath - if True then a check will be performed to ensure the
                          IR contains instructions labelled with 'fast'
         check_fastmath_result - if True then a check will be performed to
                                 ensure the result of running with fastmath
                                 on matches that of the pyfunc
         Remaining kwargs are passed to np.testing.assert_almost_equal


        Example:
            def foo():
                acc = 0
                for x in range(5):
                    for y in range(10):
                        acc +=1
                return acc

            # calling as
            prange_tester(foo)
            # will test code equivalent to
            # def foo():
            #     acc = 0
            #     for x in prange(5): # <- changed
            #         for y in prange(10): # <- changed
            #             acc +=1
            #     return acc

            # calling as
            prange_tester(foo, patch_instance=[1])
            # will test code equivalent to
            # def foo():
            #     acc = 0
            #     for x in range(5): # <- outer loop (0) unchanged
            #         for y in prange(10): # <- inner loop (1) changed
            #             acc +=1
            #     return acc

        """
        patch_instance = kwargs.pop('patch_instance', None)
        check_fastmath = kwargs.pop('check_fastmath', False)
        check_fastmath_result = kwargs.pop('check_fastmath_result', False)
        pfunc = self.generate_prange_func(pyfunc, patch_instance)
        sig = tuple([numba.typeof(x) for x in args])
        cfunc = self.compile_njit(pyfunc, sig)
        with warnings.catch_warnings(record=True) as raised_warnings:
            warnings.simplefilter('always')
            cpfunc = self.compile_parallel(pfunc, sig)
        if check_fastmath:
            self.assert_fastmath(pfunc, sig)
        if check_fastmath_result:
            fastcpfunc = self.compile_parallel_fastmath(pfunc, sig)
            kwargs = dict({'fastmath_pcres': fastcpfunc}, **kwargs)
        self.check_parfors_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)
        return raised_warnings