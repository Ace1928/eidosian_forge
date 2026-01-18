import math
import textwrap
import sys
import pytest
import threading
import traceback
import time
import numpy as np
from numpy.testing import IS_PYPY
from . import util
class TestF77Callback(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'callback', 'foo.f')]

    @pytest.mark.parametrize('name', 't,t2'.split(','))
    def test_all(self, name):
        self.check_function(name)

    @pytest.mark.xfail(IS_PYPY, reason='PyPy cannot modify tp_doc after PyType_Ready')
    def test_docstring(self):
        expected = textwrap.dedent('        a = t(fun,[fun_extra_args])\n\n        Wrapper for ``t``.\n\n        Parameters\n        ----------\n        fun : call-back function\n\n        Other Parameters\n        ----------------\n        fun_extra_args : input tuple, optional\n            Default: ()\n\n        Returns\n        -------\n        a : int\n\n        Notes\n        -----\n        Call-back functions::\n\n            def fun(): return a\n            Return objects:\n                a : int\n        ')
        assert self.module.t.__doc__ == expected

    def check_function(self, name):
        t = getattr(self.module, name)
        r = t(lambda: 4)
        assert r == 4
        r = t(lambda a: 5, fun_extra_args=(6,))
        assert r == 5
        r = t(lambda a: a, fun_extra_args=(6,))
        assert r == 6
        r = t(lambda a: 5 + a, fun_extra_args=(7,))
        assert r == 12
        r = t(lambda a: math.degrees(a), fun_extra_args=(math.pi,))
        assert r == 180
        r = t(math.degrees, fun_extra_args=(math.pi,))
        assert r == 180
        r = t(self.module.func, fun_extra_args=(6,))
        assert r == 17
        r = t(self.module.func0)
        assert r == 11
        r = t(self.module.func0._cpointer)
        assert r == 11

        class A:

            def __call__(self):
                return 7

            def mth(self):
                return 9
        a = A()
        r = t(a)
        assert r == 7
        r = t(a.mth)
        assert r == 9

    @pytest.mark.skipif(sys.platform == 'win32', reason='Fails with MinGW64 Gfortran (Issue #9673)')
    def test_string_callback(self):

        def callback(code):
            if code == 'r':
                return 0
            else:
                return 1
        f = getattr(self.module, 'string_callback')
        r = f(callback)
        assert r == 0

    @pytest.mark.skipif(sys.platform == 'win32', reason='Fails with MinGW64 Gfortran (Issue #9673)')
    def test_string_callback_array(self):
        cu1 = np.zeros((1,), 'S8')
        cu2 = np.zeros((1, 8), 'c')
        cu3 = np.array([''], 'S8')

        def callback(cu, lencu):
            if cu.shape != (lencu,):
                return 1
            if cu.dtype != 'S8':
                return 2
            if not np.all(cu == b''):
                return 3
            return 0
        f = getattr(self.module, 'string_callback_array')
        for cu in [cu1, cu2, cu3]:
            res = f(callback, cu, cu.size)
            assert res == 0

    def test_threadsafety(self):
        errors = []

        def cb():
            time.sleep(0.001)
            r = self.module.t(lambda: 123)
            assert r == 123
            return 42

        def runner(name):
            try:
                for j in range(50):
                    r = self.module.t(cb)
                    assert r == 42
                    self.check_function(name)
            except Exception:
                errors.append(traceback.format_exc())
        threads = [threading.Thread(target=runner, args=(arg,)) for arg in ('t', 't2') for n in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        errors = '\n\n'.join(errors)
        if errors:
            raise AssertionError(errors)

    def test_hidden_callback(self):
        try:
            self.module.hidden_callback(2)
        except Exception as msg:
            assert str(msg).startswith('Callback global_f not defined')
        try:
            self.module.hidden_callback2(2)
        except Exception as msg:
            assert str(msg).startswith('cb: Callback global_f not defined')
        self.module.global_f = lambda x: x + 1
        r = self.module.hidden_callback(2)
        assert r == 3
        self.module.global_f = lambda x: x + 2
        r = self.module.hidden_callback(2)
        assert r == 4
        del self.module.global_f
        try:
            self.module.hidden_callback(2)
        except Exception as msg:
            assert str(msg).startswith('Callback global_f not defined')
        self.module.global_f = lambda x=0: x + 3
        r = self.module.hidden_callback(2)
        assert r == 5
        r = self.module.hidden_callback2(2)
        assert r == 3