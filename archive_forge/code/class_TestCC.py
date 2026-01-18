import contextlib
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from unittest import skip
from ctypes import *
import numpy as np
import llvmlite.binding as ll
from numba.core import utils
from numba.tests.support import (TestCase, tag, import_dynamic, temp_directory,
import unittest
@needs_setuptools
class TestCC(BasePYCCTest):

    def setUp(self):
        super(TestCC, self).setUp()
        self.skip_if_no_external_compiler()
        from numba.tests import compile_with_pycc
        self._test_module = compile_with_pycc
        importlib.reload(self._test_module)

    @contextlib.contextmanager
    def check_cc_compiled(self, cc):
        cc.output_dir = self.tmpdir
        cc.compile()
        with self.check_c_ext(self.tmpdir, cc.name) as lib:
            yield lib

    def check_cc_compiled_in_subprocess(self, lib, code):
        prolog = "if 1:\n            import sys\n            import types\n            # to disable numba package\n            sys.modules['numba'] = types.ModuleType('numba')\n            try:\n                from numba import njit\n            except ImportError:\n                pass\n            else:\n                raise RuntimeError('cannot disable numba package')\n\n            sys.path.insert(0, %(path)r)\n            import %(name)s as lib\n            " % {'name': lib.__name__, 'path': os.path.dirname(lib.__file__)}
        code = prolog.strip(' ') + code
        subprocess.check_call([sys.executable, '-c', code])

    def test_cc_properties(self):
        cc = self._test_module.cc
        self.assertEqual(cc.name, 'pycc_test_simple')
        d = self._test_module.cc.output_dir
        self.assertTrue(os.path.isdir(d), d)
        f = self._test_module.cc.output_file
        self.assertFalse(os.path.exists(f), f)
        self.assertTrue(os.path.basename(f).startswith('pycc_test_simple.'), f)
        if sys.platform.startswith('linux'):
            self.assertTrue(f.endswith('.so'), f)
            from numba.pycc.platform import find_pyext_ending
            self.assertIn(find_pyext_ending(), f)

    def test_compile(self):
        with self.check_cc_compiled(self._test_module.cc) as lib:
            res = lib.multi(123, 321)
            self.assertPreciseEqual(res, 123 * 321)
            res = lib.multf(987, 321)
            self.assertPreciseEqual(res, 987.0 * 321.0)
            res = lib.square(5)
            self.assertPreciseEqual(res, 25)
            self.assertIs(lib.get_none(), None)
            with self.assertRaises(ZeroDivisionError):
                lib.div(1, 0)

    def check_compile_for_cpu(self, cpu_name):
        cc = self._test_module.cc
        cc.target_cpu = cpu_name
        with self.check_cc_compiled(cc) as lib:
            res = lib.multi(123, 321)
            self.assertPreciseEqual(res, 123 * 321)
            self.assertEqual(lib.multi.__module__, 'pycc_test_simple')

    def test_compile_for_cpu(self):
        self.check_compile_for_cpu(ll.get_host_cpu_name())

    def test_compile_for_cpu_host(self):
        self.check_compile_for_cpu('host')

    def test_compile_helperlib(self):
        with self.check_cc_compiled(self._test_module.cc_helperlib) as lib:
            res = lib.power(2, 7)
            self.assertPreciseEqual(res, 128)
            for val in (-1, -1 + 0j, np.complex128(-1)):
                res = lib.sqrt(val)
                self.assertPreciseEqual(res, 1j)
            for val in (4, 4.0, np.float64(4)):
                res = lib.np_sqrt(val)
                self.assertPreciseEqual(res, 2.0)
            res = lib.spacing(1.0)
            self.assertPreciseEqual(res, 2 ** (-52))
            self.assertNotEqual(lib.random(-1), lib.random(-1))
            res = lib.random(42)
            expected = np.random.RandomState(42).random_sample()
            self.assertPreciseEqual(res, expected)
            res = lib.size(np.float64([0] * 3))
            self.assertPreciseEqual(res, 3)
            code = 'if 1:\n                from numpy.testing import assert_equal, assert_allclose\n                res = lib.power(2, 7)\n                assert res == 128\n                res = lib.random(42)\n                assert_allclose(res, %(expected)s)\n                res = lib.spacing(1.0)\n                assert_allclose(res, 2**-52)\n                ' % {'expected': expected}
            self.check_cc_compiled_in_subprocess(lib, code)

    def test_compile_nrt(self):
        with self.check_cc_compiled(self._test_module.cc_nrt) as lib:
            self.assertPreciseEqual(lib.zero_scalar(1), 0.0)
            res = lib.zeros(3)
            self.assertEqual(list(res), [0, 0, 0])
            if has_blas:
                res = lib.vector_dot(4)
                self.assertPreciseEqual(res, 30.0)
            val = np.float64([2.0, 5.0, 1.0, 3.0, 4.0])
            res = lib.np_argsort(val)
            expected = np.argsort(val)
            self.assertPreciseEqual(res, expected)
            code = 'if 1:\n                from numpy.testing import assert_equal\n                from numpy import float64, argsort\n                res = lib.zero_scalar(1)\n                assert res == 0.0\n                res = lib.zeros(3)\n                assert list(res) == [0, 0, 0]\n                if %(has_blas)s:\n                    res = lib.vector_dot(4)\n                    assert res == 30.0\n                val = float64([2., 5., 1., 3., 4.])\n                res = lib.np_argsort(val)\n                expected = argsort(val)\n                assert_equal(res, expected)\n                ' % dict(has_blas=has_blas)
            self.check_cc_compiled_in_subprocess(lib, code)

    def test_hashing(self):
        with self.check_cc_compiled(self._test_module.cc_nrt) as lib:
            res = lib.hash_literal_str_A()
            self.assertPreciseEqual(res, hash('A'))
            res = lib.hash_str('A')
            self.assertPreciseEqual(res, hash('A'))
            code = 'if 1:\n                from numpy.testing import assert_equal\n                res = lib.hash_literal_str_A()\n                assert_equal(res, hash("A"))\n                res = lib.hash_str("A")\n                assert_equal(res, hash("A"))\n                '
            self.check_cc_compiled_in_subprocess(lib, code)

    def test_c_extension_usecase(self):
        with self.check_cc_compiled(self._test_module.cc_nrt) as lib:
            arr = np.arange(128, dtype=np.intp)
            got = lib.dict_usecase(arr)
            expect = arr * arr
            self.assertPreciseEqual(got, expect)