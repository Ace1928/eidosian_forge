import sys
import platform
import pytest
import numpy as np
import numpy.core._multiarray_umath as ncu
from numpy.testing import (
class TestComplexAbsoluteMixedDTypes:

    @pytest.mark.parametrize('stride', [-4, -3, -2, -1, 1, 2, 3, 4])
    @pytest.mark.parametrize('astype', [np.complex64, np.complex128])
    @pytest.mark.parametrize('func', ['abs', 'square', 'conjugate'])
    def test_array(self, stride, astype, func):
        dtype = [('template_id', '<i8'), ('bank_chisq', '<f4'), ('bank_chisq_dof', '<i8'), ('chisq', '<f4'), ('chisq_dof', '<i8'), ('cont_chisq', '<f4'), ('psd_var_val', '<f4'), ('sg_chisq', '<f4'), ('mycomplex', astype), ('time_index', '<i8')]
        vec = np.array([(0, 0.0, 0, -31.666483, 200, 0.0, 0.0, 1.0, 3.0 + 4j, 613090), (1, 0.0, 0, 260.91525, 42, 0.0, 0.0, 1.0, 5.0 + 12j, 787315), (1, 0.0, 0, 52.15155, 42, 0.0, 0.0, 1.0, 8.0 + 15j, 806641), (1, 0.0, 0, 52.430195, 42, 0.0, 0.0, 1.0, 7.0 + 24j, 1363540), (2, 0.0, 0, 304.43646, 58, 0.0, 0.0, 1.0, 20.0 + 21j, 787323), (3, 0.0, 0, 299.42108, 52, 0.0, 0.0, 1.0, 12.0 + 35j, 787332), (4, 0.0, 0, 39.4836, 28, 0.0, 0.0, 9.182192, 9.0 + 40j, 787304), (4, 0.0, 0, 76.83787, 28, 0.0, 0.0, 1.0, 28.0 + 45j, 1321869), (5, 0.0, 0, 143.26366, 24, 0.0, 0.0, 10.996129, 11.0 + 60j, 787299)], dtype=dtype)
        myfunc = getattr(np, func)
        a = vec['mycomplex']
        g = myfunc(a[::stride])
        b = vec['mycomplex'].copy()
        h = myfunc(b[::stride])
        assert_array_max_ulp(h.real, g.real, 1)
        assert_array_max_ulp(h.imag, g.imag, 1)