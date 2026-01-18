import pytest
from numpy import (
from numpy.testing import (
class TestLogspace:

    def test_basic(self):
        y = logspace(0, 6)
        assert_(len(y) == 50)
        y = logspace(0, 6, num=100)
        assert_(y[-1] == 10 ** 6)
        y = logspace(0, 6, endpoint=False)
        assert_(y[-1] < 10 ** 6)
        y = logspace(0, 6, num=7)
        assert_array_equal(y, [1, 10, 100, 1000.0, 10000.0, 100000.0, 1000000.0])

    def test_start_stop_array(self):
        start = array([0.0, 1.0])
        stop = array([6.0, 7.0])
        t1 = logspace(start, stop, 6)
        t2 = stack([logspace(_start, _stop, 6) for _start, _stop in zip(start, stop)], axis=1)
        assert_equal(t1, t2)
        t3 = logspace(start, stop[0], 6)
        t4 = stack([logspace(_start, stop[0], 6) for _start in start], axis=1)
        assert_equal(t3, t4)
        t5 = logspace(start, stop, 6, axis=-1)
        assert_equal(t5, t2.T)

    @pytest.mark.parametrize('axis', [0, 1, -1])
    def test_base_array(self, axis: int):
        start = 1
        stop = 2
        num = 6
        base = array([1, 2])
        t1 = logspace(start, stop, num=num, base=base, axis=axis)
        t2 = stack([logspace(start, stop, num=num, base=_base) for _base in base], axis=(axis + 1) % t1.ndim)
        assert_equal(t1, t2)

    @pytest.mark.parametrize('axis', [0, 1, -1])
    def test_stop_base_array(self, axis: int):
        start = 1
        stop = array([2, 3])
        num = 6
        base = array([1, 2])
        t1 = logspace(start, stop, num=num, base=base, axis=axis)
        t2 = stack([logspace(start, _stop, num=num, base=_base) for _stop, _base in zip(stop, base)], axis=(axis + 1) % t1.ndim)
        assert_equal(t1, t2)

    def test_dtype(self):
        y = logspace(0, 6, dtype='float32')
        assert_equal(y.dtype, dtype('float32'))
        y = logspace(0, 6, dtype='float64')
        assert_equal(y.dtype, dtype('float64'))
        y = logspace(0, 6, dtype='int32')
        assert_equal(y.dtype, dtype('int32'))

    def test_physical_quantities(self):
        a = PhysicalQuantity(1.0)
        b = PhysicalQuantity(5.0)
        assert_equal(logspace(a, b), logspace(1.0, 5.0))

    def test_subclass(self):
        a = array(1).view(PhysicalQuantity2)
        b = array(7).view(PhysicalQuantity2)
        ls = logspace(a, b)
        assert type(ls) is PhysicalQuantity2
        assert_equal(ls, logspace(1.0, 7.0))
        ls = logspace(a, b, 1)
        assert type(ls) is PhysicalQuantity2
        assert_equal(ls, logspace(1.0, 7.0, 1))