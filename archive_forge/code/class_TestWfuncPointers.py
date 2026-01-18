import numpy as np
from numpy.testing import (assert_allclose,
import scipy.linalg.cython_blas as blas
class TestWfuncPointers:
    """ Test the function pointers that are expected to fail on
    Mac OS X without the additional entry statement in their definitions
    in fblas_l1.pyf.src. """

    def test_complex_args(self):
        cx = np.array([0.5 + 1j, 0.25 - 0.375j, 12.5 - 4j], np.complex64)
        cy = np.array([0.8 + 2j, 0.875 - 0.625j, -1.0 + 2j], np.complex64)
        assert_allclose(blas._test_cdotc(cx, cy), -17.6468753815 + 21.3718757629j)
        assert_allclose(blas._test_cdotu(cx, cy), -6.11562538147 + 30.3156242371j)
        assert_equal(blas._test_icamax(cx), 3)
        assert_allclose(blas._test_scasum(cx), 18.625)
        assert_allclose(blas._test_scnrm2(cx), 13.1796483994)
        assert_allclose(blas._test_cdotc(cx[::2], cy[::2]), -18.1000003815 + 21.2000007629j)
        assert_allclose(blas._test_cdotu(cx[::2], cy[::2]), -6.10000038147 + 30.7999992371j)
        assert_allclose(blas._test_scasum(cx[::2]), 18.0)
        assert_allclose(blas._test_scnrm2(cx[::2]), 13.1719398499)

    def test_double_args(self):
        x = np.array([5.0, -3, -0.5], np.float64)
        y = np.array([2, 1, 0.5], np.float64)
        assert_allclose(blas._test_dasum(x), 8.5)
        assert_allclose(blas._test_ddot(x, y), 6.75)
        assert_allclose(blas._test_dnrm2(x), 5.85234975815)
        assert_allclose(blas._test_dasum(x[::2]), 5.5)
        assert_allclose(blas._test_ddot(x[::2], y[::2]), 9.75)
        assert_allclose(blas._test_dnrm2(x[::2]), 5.0249376297)
        assert_equal(blas._test_idamax(x), 1)

    def test_float_args(self):
        x = np.array([5.0, -3, -0.5], np.float32)
        y = np.array([2, 1, 0.5], np.float32)
        assert_equal(blas._test_isamax(x), 1)
        assert_allclose(blas._test_sasum(x), 8.5)
        assert_allclose(blas._test_sdot(x, y), 6.75)
        assert_allclose(blas._test_snrm2(x), 5.85234975815)
        assert_allclose(blas._test_sasum(x[::2]), 5.5)
        assert_allclose(blas._test_sdot(x[::2], y[::2]), 9.75)
        assert_allclose(blas._test_snrm2(x[::2]), 5.0249376297)

    def test_double_complex_args(self):
        cx = np.array([0.5 + 1j, 0.25 - 0.375j, 13.0 - 4j], np.complex128)
        cy = np.array([0.875 + 2j, 0.875 - 0.625j, -1.0 + 2j], np.complex128)
        assert_equal(blas._test_izamax(cx), 3)
        assert_allclose(blas._test_zdotc(cx, cy), -18.109375 + 22.296875j)
        assert_allclose(blas._test_zdotu(cx, cy), -6.578125 + 31.390625j)
        assert_allclose(blas._test_zdotc(cx[::2], cy[::2]), -18.5625 + 22.125j)
        assert_allclose(blas._test_zdotu(cx[::2], cy[::2]), -6.5625 + 31.875j)