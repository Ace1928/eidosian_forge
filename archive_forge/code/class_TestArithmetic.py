from functools import reduce
import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
class TestArithmetic:
    x = np.linspace(-1, 1, 100)

    def test_legadd(self):
        for i in range(5):
            for j in range(5):
                msg = f'At i={i}, j={j}'
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = leg.legadd([0] * i + [1], [0] * j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_legsub(self):
        for i in range(5):
            for j in range(5):
                msg = f'At i={i}, j={j}'
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = leg.legsub([0] * i + [1], [0] * j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_legmulx(self):
        assert_equal(leg.legmulx([0]), [0])
        assert_equal(leg.legmulx([1]), [0, 1])
        for i in range(1, 5):
            tmp = 2 * i + 1
            ser = [0] * i + [1]
            tgt = [0] * (i - 1) + [i / tmp, 0, (i + 1) / tmp]
            assert_equal(leg.legmulx(ser), tgt)

    def test_legmul(self):
        for i in range(5):
            pol1 = [0] * i + [1]
            val1 = leg.legval(self.x, pol1)
            for j in range(5):
                msg = f'At i={i}, j={j}'
                pol2 = [0] * j + [1]
                val2 = leg.legval(self.x, pol2)
                pol3 = leg.legmul(pol1, pol2)
                val3 = leg.legval(self.x, pol3)
                assert_(len(pol3) == i + j + 1, msg)
                assert_almost_equal(val3, val1 * val2, err_msg=msg)

    def test_legdiv(self):
        for i in range(5):
            for j in range(5):
                msg = f'At i={i}, j={j}'
                ci = [0] * i + [1]
                cj = [0] * j + [1]
                tgt = leg.legadd(ci, cj)
                quo, rem = leg.legdiv(tgt, ci)
                res = leg.legadd(leg.legmul(quo, ci), rem)
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_legpow(self):
        for i in range(5):
            for j in range(5):
                msg = f'At i={i}, j={j}'
                c = np.arange(i + 1)
                tgt = reduce(leg.legmul, [c] * j, np.array([1]))
                res = leg.legpow(c, j)
                assert_equal(trim(res), trim(tgt), err_msg=msg)