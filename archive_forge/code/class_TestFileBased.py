import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
class TestFileBased:
    ldbl = 1 + LD_INFO.eps
    tgt = np.array([ldbl] * 5)
    out = ''.join([repr(t) + '\n' for t in tgt])

    def test_fromfile_bogus(self):
        with temppath() as path:
            with open(path, 'w') as f:
                f.write('1. 2. 3. flop 4.\n')
            with assert_warns(DeprecationWarning):
                res = np.fromfile(path, dtype=float, sep=' ')
        assert_equal(res, np.array([1.0, 2.0, 3.0]))

    def test_fromfile_complex(self):
        for ctype in ['complex', 'cdouble', 'cfloat']:
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write('1, 2 ,  3  ,4\n')
                res = np.fromfile(path, dtype=ctype, sep=',')
            assert_equal(res, np.array([1.0, 2.0, 3.0, 4.0]))
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write('1j, -2j,  3j, 4e1j\n')
                res = np.fromfile(path, dtype=ctype, sep=',')
            assert_equal(res, np.array([1j, -2j, 3j, 40j]))
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write('1+1j,2-2j, -3+3j,  -4e1+4j\n')
                res = np.fromfile(path, dtype=ctype, sep=',')
            assert_equal(res, np.array([1.0 + 1j, 2.0 - 2j, -3.0 + 3j, -40.0 + 4j]))
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write('1+2 j,3\n')
                with assert_warns(DeprecationWarning):
                    res = np.fromfile(path, dtype=ctype, sep=',')
            assert_equal(res, np.array([1.0]))
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write('1+ 2j,3\n')
                with assert_warns(DeprecationWarning):
                    res = np.fromfile(path, dtype=ctype, sep=',')
            assert_equal(res, np.array([1.0]))
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write('1 +2j,3\n')
                with assert_warns(DeprecationWarning):
                    res = np.fromfile(path, dtype=ctype, sep=',')
            assert_equal(res, np.array([1.0]))
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write('1+j\n')
                with assert_warns(DeprecationWarning):
                    res = np.fromfile(path, dtype=ctype, sep=',')
            assert_equal(res, np.array([1.0]))
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write('1+\n')
                with assert_warns(DeprecationWarning):
                    res = np.fromfile(path, dtype=ctype, sep=',')
            assert_equal(res, np.array([1.0]))
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write('1j+1\n')
                with assert_warns(DeprecationWarning):
                    res = np.fromfile(path, dtype=ctype, sep=',')
            assert_equal(res, np.array([1j]))

    @pytest.mark.skipif(string_to_longdouble_inaccurate, reason='Need strtold_l')
    def test_fromfile(self):
        with temppath() as path:
            with open(path, 'w') as f:
                f.write(self.out)
            res = np.fromfile(path, dtype=np.longdouble, sep='\n')
        assert_equal(res, self.tgt)

    @pytest.mark.skipif(string_to_longdouble_inaccurate, reason='Need strtold_l')
    def test_genfromtxt(self):
        with temppath() as path:
            with open(path, 'w') as f:
                f.write(self.out)
            res = np.genfromtxt(path, dtype=np.longdouble)
        assert_equal(res, self.tgt)

    @pytest.mark.skipif(string_to_longdouble_inaccurate, reason='Need strtold_l')
    def test_loadtxt(self):
        with temppath() as path:
            with open(path, 'w') as f:
                f.write(self.out)
            res = np.loadtxt(path, dtype=np.longdouble)
        assert_equal(res, self.tgt)

    @pytest.mark.skipif(string_to_longdouble_inaccurate, reason='Need strtold_l')
    def test_tofile_roundtrip(self):
        with temppath() as path:
            self.tgt.tofile(path, sep=' ')
            res = np.fromfile(path, dtype=np.longdouble, sep=' ')
        assert_equal(res, self.tgt)