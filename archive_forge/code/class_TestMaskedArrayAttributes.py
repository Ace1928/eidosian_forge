import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
class TestMaskedArrayAttributes:

    def test_keepmask(self):
        x = masked_array([1, 2, 3], mask=[1, 0, 0])
        mx = masked_array(x)
        assert_equal(mx.mask, x.mask)
        mx = masked_array(x, mask=[0, 1, 0], keep_mask=False)
        assert_equal(mx.mask, [0, 1, 0])
        mx = masked_array(x, mask=[0, 1, 0], keep_mask=True)
        assert_equal(mx.mask, [1, 1, 0])
        mx = masked_array(x, mask=[0, 1, 0])
        assert_equal(mx.mask, [1, 1, 0])

    def test_hardmask(self):
        d = arange(5)
        n = [0, 0, 0, 1, 1]
        m = make_mask(n)
        xh = array(d, mask=m, hard_mask=True)
        xs = array(d, mask=m, hard_mask=False, copy=True)
        xh[[1, 4]] = [10, 40]
        xs[[1, 4]] = [10, 40]
        assert_equal(xh._data, [0, 10, 2, 3, 4])
        assert_equal(xs._data, [0, 10, 2, 3, 40])
        assert_equal(xs.mask, [0, 0, 0, 1, 0])
        assert_(xh._hardmask)
        assert_(not xs._hardmask)
        xh[1:4] = [10, 20, 30]
        xs[1:4] = [10, 20, 30]
        assert_equal(xh._data, [0, 10, 20, 3, 4])
        assert_equal(xs._data, [0, 10, 20, 30, 40])
        assert_equal(xs.mask, nomask)
        xh[0] = masked
        xs[0] = masked
        assert_equal(xh.mask, [1, 0, 0, 1, 1])
        assert_equal(xs.mask, [1, 0, 0, 0, 0])
        xh[:] = 1
        xs[:] = 1
        assert_equal(xh._data, [0, 1, 1, 3, 4])
        assert_equal(xs._data, [1, 1, 1, 1, 1])
        assert_equal(xh.mask, [1, 0, 0, 1, 1])
        assert_equal(xs.mask, nomask)
        xh.soften_mask()
        xh[:] = arange(5)
        assert_equal(xh._data, [0, 1, 2, 3, 4])
        assert_equal(xh.mask, nomask)
        xh.harden_mask()
        xh[xh < 3] = masked
        assert_equal(xh._data, [0, 1, 2, 3, 4])
        assert_equal(xh._mask, [1, 1, 1, 0, 0])
        xh[filled(xh > 1, False)] = 5
        assert_equal(xh._data, [0, 1, 2, 5, 5])
        assert_equal(xh._mask, [1, 1, 1, 0, 0])
        xh = array([[1, 2], [3, 4]], mask=[[1, 0], [0, 0]], hard_mask=True)
        xh[0] = 0
        assert_equal(xh._data, [[1, 0], [3, 4]])
        assert_equal(xh._mask, [[1, 0], [0, 0]])
        xh[-1, -1] = 5
        assert_equal(xh._data, [[1, 0], [3, 5]])
        assert_equal(xh._mask, [[1, 0], [0, 0]])
        xh[filled(xh < 5, False)] = 2
        assert_equal(xh._data, [[1, 2], [2, 5]])
        assert_equal(xh._mask, [[1, 0], [0, 0]])

    def test_hardmask_again(self):
        d = arange(5)
        n = [0, 0, 0, 1, 1]
        m = make_mask(n)
        xh = array(d, mask=m, hard_mask=True)
        xh[4:5] = 999
        xh[0:1] = 999
        assert_equal(xh._data, [999, 1, 2, 3, 4])

    def test_hardmask_oncemore_yay(self):
        a = array([1, 2, 3], mask=[1, 0, 0])
        b = a.harden_mask()
        assert_equal(a, b)
        b[0] = 0
        assert_equal(a, b)
        assert_equal(b, array([1, 2, 3], mask=[1, 0, 0]))
        a = b.soften_mask()
        a[0] = 0
        assert_equal(a, b)
        assert_equal(b, array([0, 2, 3], mask=[0, 0, 0]))

    def test_smallmask(self):
        a = arange(10)
        a[1] = masked
        a[1] = 1
        assert_equal(a._mask, nomask)
        a = arange(10)
        a._smallmask = False
        a[1] = masked
        a[1] = 1
        assert_equal(a._mask, zeros(10))

    def test_shrink_mask(self):
        a = array([1, 2, 3], mask=[0, 0, 0])
        b = a.shrink_mask()
        assert_equal(a, b)
        assert_equal(a.mask, nomask)
        a = np.ma.array([(1, 2.0)], [('a', int), ('b', float)])
        b = a.copy()
        a.shrink_mask()
        assert_equal(a.mask, b.mask)

    def test_flat(self):
        x = array([[(1, 1.1, 'one'), (2, 2.2, 'two'), (3, 3.3, 'thr')], [(4, 4.4, 'fou'), (5, 5.5, 'fiv'), (6, 6.6, 'six')]], dtype=[('a', int), ('b', float), ('c', '|S8')])
        x['a'][0, 1] = masked
        x['b'][1, 0] = masked
        x['c'][0, 2] = masked
        x[-1, -1] = masked
        xflat = x.flat
        assert_equal(xflat[0], x[0, 0])
        assert_equal(xflat[1], x[0, 1])
        assert_equal(xflat[2], x[0, 2])
        assert_equal(xflat[:3], x[0])
        assert_equal(xflat[3], x[1, 0])
        assert_equal(xflat[4], x[1, 1])
        assert_equal(xflat[5], x[1, 2])
        assert_equal(xflat[3:], x[1])
        assert_equal(xflat[-1], x[-1, -1])
        i = 0
        j = 0
        for xf in xflat:
            assert_equal(xf, x[j, i])
            i += 1
            if i >= x.shape[-1]:
                i = 0
                j += 1

    def test_assign_dtype(self):
        a = np.zeros(4, dtype='f4,i4')
        m = np.ma.array(a)
        m.dtype = np.dtype('f4')
        repr(m)
        assert_equal(m.dtype, np.dtype('f4'))

        def assign():
            m = np.ma.array(a)
            m.dtype = np.dtype('f8')
        assert_raises(ValueError, assign)
        b = a.view(dtype='f4', type=np.ma.MaskedArray)
        assert_equal(b.dtype, np.dtype('f4'))
        a = np.zeros(4, dtype='f4')
        m = np.ma.array(a)
        m.dtype = np.dtype('f4,i4')
        assert_equal(m.dtype, np.dtype('f4,i4'))
        assert_equal(m._mask, np.ma.nomask)