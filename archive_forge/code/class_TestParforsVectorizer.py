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
@x86_only
class TestParforsVectorizer(TestPrangeBase):
    _numba_parallel_test_ = False

    def get_gufunc_asm(self, func, schedule_type, *args, **kwargs):
        fastmath = kwargs.pop('fastmath', False)
        cpu_name = kwargs.pop('cpu_name', 'skylake-avx512')
        assertions = kwargs.pop('assertions', True)
        cpu_features = kwargs.pop('cpu_features', '-prefer-256-bit')
        env_opts = {'NUMBA_CPU_NAME': cpu_name, 'NUMBA_CPU_FEATURES': cpu_features}
        overrides = []
        for k, v in env_opts.items():
            overrides.append(override_env_config(k, v))
        with overrides[0], overrides[1]:
            sig = tuple([numba.typeof(x) for x in args])
            pfunc_vectorizable = self.generate_prange_func(func, None)
            if fastmath == True:
                cres = self.compile_parallel_fastmath(pfunc_vectorizable, sig)
            else:
                cres = self.compile_parallel(pfunc_vectorizable, sig)
            asm = self._get_gufunc_asm(cres)
            if assertions:
                schedty = re.compile('call\\s+\\w+\\*\\s+@do_scheduling_(\\w+)\\(')
                matches = schedty.findall(cres.library.get_llvm_str())
                self.assertGreaterEqual(len(matches), 1)
                self.assertEqual(matches[0], schedule_type)
                self.assertNotEqual(asm, {})
            return asm

    @linux_only
    @TestCase.run_test_in_subprocess
    def test_vectorizer_fastmath_asm(self):
        """ This checks that if fastmath is set and the underlying hardware
        is suitable, and the function supplied is amenable to fastmath based
        vectorization, that the vectorizer actually runs.
        """

        def will_vectorize(A):
            n = len(A)
            acc = 0
            for i in range(n):
                acc += np.sqrt(i)
            return acc
        arg = np.zeros(10)
        fast_asm = self.get_gufunc_asm(will_vectorize, 'unsigned', arg, fastmath=True)
        slow_asm = self.get_gufunc_asm(will_vectorize, 'unsigned', arg, fastmath=False)
        for v in fast_asm.values():
            self.assertTrue('vaddpd' in v)
            self.assertTrue('vsqrtpd' in v or '__svml_sqrt' in v)
            self.assertTrue('zmm' in v)
        for v in slow_asm.values():
            self.assertTrue('vaddpd' not in v)
            self.assertTrue('vsqrtpd' not in v)
            self.assertTrue('vsqrtsd' in v and '__svml_sqrt' not in v)
            self.assertTrue('vaddsd' in v)
            self.assertTrue('zmm' not in v)

    @linux_only
    @TestCase.run_test_in_subprocess(envvars={'NUMBA_BOUNDSCHECK': '0'})
    def test_unsigned_refusal_to_vectorize(self):
        """ This checks that if fastmath is set and the underlying hardware
        is suitable, and the function supplied is amenable to fastmath based
        vectorization, that the vectorizer actually runs.
        """

        def will_not_vectorize(A):
            n = len(A)
            for i in range(-n, 0):
                A[i] = np.sqrt(A[i])
            return A

        def will_vectorize(A):
            n = len(A)
            for i in range(n):
                A[i] = np.sqrt(A[i])
            return A
        arg = np.zeros(10)
        self.assertFalse(config.BOUNDSCHECK)
        novec_asm = self.get_gufunc_asm(will_not_vectorize, 'signed', arg, fastmath=True)
        vec_asm = self.get_gufunc_asm(will_vectorize, 'unsigned', arg, fastmath=True)
        for v in novec_asm.values():
            self.assertTrue('vsqrtpd' not in v)
            self.assertTrue('vsqrtsd' in v)
            self.assertTrue('zmm' not in v)
        for v in vec_asm.values():
            self.assertTrue('vsqrtpd' in v or '__svml_sqrt' in v)
            self.assertTrue('vmovupd' in v)
            self.assertTrue('zmm' in v)

    @linux_only
    @TestCase.run_test_in_subprocess(envvars={'NUMBA_BOUNDSCHECK': '0'})
    def test_signed_vs_unsigned_vec_asm(self):
        """ This checks vectorization for signed vs unsigned variants of a
        trivial accumulator, the only meaningful difference should be the
        presence of signed vs. unsigned unpack instructions (for the
        induction var).
        """

        def signed_variant():
            n = 4096
            A = 0.0
            for i in range(-n, 0):
                A += i
            return A

        def unsigned_variant():
            n = 4096
            A = 0.0
            for i in range(n):
                A += i
            return A
        self.assertFalse(config.BOUNDSCHECK)
        signed_asm = self.get_gufunc_asm(signed_variant, 'signed', fastmath=True)
        unsigned_asm = self.get_gufunc_asm(unsigned_variant, 'unsigned', fastmath=True)

        def strip_instrs(asm):
            acc = []
            for x in asm.splitlines():
                spd = x.strip()
                if spd != '' and (not (spd.startswith('.') or spd.startswith('_') or spd.startswith('"') or ('__numba_parfor_gufunc' in spd))):
                    acc.append(re.sub('[\t]', '', spd))
            return acc
        for k, v in signed_asm.items():
            signed_instr = strip_instrs(v)
            break
        for k, v in unsigned_asm.items():
            unsigned_instr = strip_instrs(v)
            break
        from difflib import SequenceMatcher as sm
        self.assertEqual(len(signed_instr), len(unsigned_instr))
        for a, b in zip(signed_instr, unsigned_instr):
            if a == b:
                continue
            else:
                s = sm(lambda x: x == '\t', a, b)
                ops = s.get_opcodes()
                for op in ops:
                    if op[0] == 'insert':
                        self.assertEqual(b[op[-2]:op[-1]], 'u')