import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.testing import assert_, assert_raises
from numpy.ma.testutils import assert_equal
from numpy.ma.core import (
class TestClassWrapping:

    def setup_method(self):
        m = np.ma.masked_array([1, 3, 5], mask=[False, True, False])
        wm = WrappedArray(m)
        self.data = (m, wm)

    def test_masked_unary_operations(self):
        m, wm = self.data
        with np.errstate(divide='ignore'):
            assert_(isinstance(np.log(wm), WrappedArray))

    def test_masked_binary_operations(self):
        m, wm = self.data
        assert_(isinstance(np.add(wm, wm), WrappedArray))
        assert_(isinstance(np.add(m, wm), WrappedArray))
        assert_(isinstance(np.add(wm, m), WrappedArray))
        assert_equal(np.add(m, wm), m + wm)
        assert_(isinstance(np.hypot(m, wm), WrappedArray))
        assert_(isinstance(np.hypot(wm, m), WrappedArray))
        assert_(isinstance(np.divide(wm, m), WrappedArray))
        assert_(isinstance(np.divide(m, wm), WrappedArray))
        assert_equal(np.divide(wm, m) * m, np.divide(m, m) * wm)
        m2 = np.stack([m, m])
        assert_(isinstance(np.divide(wm, m2), WrappedArray))
        assert_(isinstance(np.divide(m2, wm), WrappedArray))
        assert_equal(np.divide(m2, wm), np.divide(wm, m2))

    def test_mixins_have_slots(self):
        mixin = NDArrayOperatorsMixin()
        assert_raises(AttributeError, mixin.__setattr__, 'not_a_real_attr', 1)
        m = np.ma.masked_array([1, 3, 5], mask=[False, True, False])
        wm = WrappedArray(m)
        assert_raises(AttributeError, wm.__setattr__, 'not_an_attr', 2)