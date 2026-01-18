import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
class TestSetOps:

    def test_intersect1d(self):
        a = np.array([5, 7, 1, 2])
        b = np.array([2, 4, 3, 1, 5])
        ec = np.array([1, 2, 5])
        c = intersect1d(a, b, assume_unique=True)
        assert_array_equal(c, ec)
        a = np.array([5, 5, 7, 1, 2])
        b = np.array([2, 1, 4, 3, 3, 1, 5])
        ed = np.array([1, 2, 5])
        c = intersect1d(a, b)
        assert_array_equal(c, ed)
        assert_array_equal([], intersect1d([], []))

    def test_intersect1d_array_like(self):

        class Test:

            def __array__(self):
                return np.arange(3)
        a = Test()
        res = intersect1d(a, a)
        assert_array_equal(res, a)
        res = intersect1d([1, 2, 3], [1, 2, 3])
        assert_array_equal(res, [1, 2, 3])

    def test_intersect1d_indices(self):
        a = np.array([1, 2, 3, 4])
        b = np.array([2, 1, 4, 6])
        c, i1, i2 = intersect1d(a, b, assume_unique=True, return_indices=True)
        ee = np.array([1, 2, 4])
        assert_array_equal(c, ee)
        assert_array_equal(a[i1], ee)
        assert_array_equal(b[i2], ee)
        a = np.array([1, 2, 2, 3, 4, 3, 2])
        b = np.array([1, 8, 4, 2, 2, 3, 2, 3])
        c, i1, i2 = intersect1d(a, b, return_indices=True)
        ef = np.array([1, 2, 3, 4])
        assert_array_equal(c, ef)
        assert_array_equal(a[i1], ef)
        assert_array_equal(b[i2], ef)
        a = np.array([[2, 4, 5, 6], [7, 8, 1, 15]])
        b = np.array([[3, 2, 7, 6], [10, 12, 8, 9]])
        c, i1, i2 = intersect1d(a, b, assume_unique=True, return_indices=True)
        ui1 = np.unravel_index(i1, a.shape)
        ui2 = np.unravel_index(i2, b.shape)
        ea = np.array([2, 6, 7, 8])
        assert_array_equal(ea, a[ui1])
        assert_array_equal(ea, b[ui2])
        a = np.array([[2, 4, 5, 6, 6], [4, 7, 8, 7, 2]])
        b = np.array([[3, 2, 7, 7], [10, 12, 8, 7]])
        c, i1, i2 = intersect1d(a, b, return_indices=True)
        ui1 = np.unravel_index(i1, a.shape)
        ui2 = np.unravel_index(i2, b.shape)
        ea = np.array([2, 7, 8])
        assert_array_equal(ea, a[ui1])
        assert_array_equal(ea, b[ui2])

    def test_setxor1d(self):
        a = np.array([5, 7, 1, 2])
        b = np.array([2, 4, 3, 1, 5])
        ec = np.array([3, 4, 7])
        c = setxor1d(a, b)
        assert_array_equal(c, ec)
        a = np.array([1, 2, 3])
        b = np.array([6, 5, 4])
        ec = np.array([1, 2, 3, 4, 5, 6])
        c = setxor1d(a, b)
        assert_array_equal(c, ec)
        a = np.array([1, 8, 2, 3])
        b = np.array([6, 5, 4, 8])
        ec = np.array([1, 2, 3, 4, 5, 6])
        c = setxor1d(a, b)
        assert_array_equal(c, ec)
        assert_array_equal([], setxor1d([], []))

    def test_ediff1d(self):
        zero_elem = np.array([])
        one_elem = np.array([1])
        two_elem = np.array([1, 2])
        assert_array_equal([], ediff1d(zero_elem))
        assert_array_equal([0], ediff1d(zero_elem, to_begin=0))
        assert_array_equal([0], ediff1d(zero_elem, to_end=0))
        assert_array_equal([-1, 0], ediff1d(zero_elem, to_begin=-1, to_end=0))
        assert_array_equal([], ediff1d(one_elem))
        assert_array_equal([1], ediff1d(two_elem))
        assert_array_equal([7, 1, 9], ediff1d(two_elem, to_begin=7, to_end=9))
        assert_array_equal([5, 6, 1, 7, 8], ediff1d(two_elem, to_begin=[5, 6], to_end=[7, 8]))
        assert_array_equal([1, 9], ediff1d(two_elem, to_end=9))
        assert_array_equal([1, 7, 8], ediff1d(two_elem, to_end=[7, 8]))
        assert_array_equal([7, 1], ediff1d(two_elem, to_begin=7))
        assert_array_equal([5, 6, 1], ediff1d(two_elem, to_begin=[5, 6]))

    @pytest.mark.parametrize('ary, prepend, append, expected', [(np.array([1, 2, 3], dtype=np.int64), None, np.nan, 'to_end'), (np.array([1, 2, 3], dtype=np.int64), np.array([5, 7, 2], dtype=np.float32), None, 'to_begin'), (np.array([1.0, 3.0, 9.0], dtype=np.int8), np.nan, np.nan, 'to_begin')])
    def test_ediff1d_forbidden_type_casts(self, ary, prepend, append, expected):
        msg = 'dtype of `{}` must be compatible'.format(expected)
        with assert_raises_regex(TypeError, msg):
            ediff1d(ary=ary, to_end=append, to_begin=prepend)

    @pytest.mark.parametrize('ary,prepend,append,expected', [(np.array([1, 2, 3], dtype=np.int16), 2 ** 16, 2 ** 16 + 4, np.array([0, 1, 1, 4], dtype=np.int16)), (np.array([1, 2, 3], dtype=np.float32), np.array([5], dtype=np.float64), None, np.array([5, 1, 1], dtype=np.float32)), (np.array([1, 2, 3], dtype=np.int32), 0, 0, np.array([0, 1, 1, 0], dtype=np.int32)), (np.array([1, 2, 3], dtype=np.int64), 3, -9, np.array([3, 1, 1, -9], dtype=np.int64))])
    def test_ediff1d_scalar_handling(self, ary, prepend, append, expected):
        actual = np.ediff1d(ary=ary, to_end=append, to_begin=prepend)
        assert_equal(actual, expected)
        assert actual.dtype == expected.dtype

    @pytest.mark.parametrize('kind', [None, 'sort', 'table'])
    def test_isin(self, kind):

        def _isin_slow(a, b):
            b = np.asarray(b).flatten().tolist()
            return a in b
        isin_slow = np.vectorize(_isin_slow, otypes=[bool], excluded={1})

        def assert_isin_equal(a, b):
            x = isin(a, b, kind=kind)
            y = isin_slow(a, b)
            assert_array_equal(x, y)
        a = np.arange(24).reshape([2, 3, 4])
        b = np.array([[10, 20, 30], [0, 1, 3], [11, 22, 33]])
        assert_isin_equal(a, b)
        c = [(9, 8), (7, 6)]
        d = (9, 7)
        assert_isin_equal(c, d)
        f = np.array(3)
        assert_isin_equal(f, b)
        assert_isin_equal(a, f)
        assert_isin_equal(f, f)
        assert_isin_equal(5, b)
        assert_isin_equal(a, 6)
        assert_isin_equal(5, 6)
        if kind != 'table':
            x = []
            assert_isin_equal(x, b)
            assert_isin_equal(a, x)
            assert_isin_equal(x, x)
        for dtype in [bool, np.int64, np.float64]:
            if kind == 'table' and dtype == np.float64:
                continue
            if dtype in {np.int64, np.float64}:
                ar = np.array([10, 20, 30], dtype=dtype)
            elif dtype in {bool}:
                ar = np.array([True, False, False])
            empty_array = np.array([], dtype=dtype)
            assert_isin_equal(empty_array, ar)
            assert_isin_equal(ar, empty_array)
            assert_isin_equal(empty_array, empty_array)

    @pytest.mark.parametrize('kind', [None, 'sort', 'table'])
    def test_in1d(self, kind):
        for mult in (1, 10):
            a = [5, 7, 1, 2]
            b = [2, 4, 3, 1, 5] * mult
            ec = np.array([True, False, True, True])
            c = in1d(a, b, assume_unique=True, kind=kind)
            assert_array_equal(c, ec)
            a[0] = 8
            ec = np.array([False, False, True, True])
            c = in1d(a, b, assume_unique=True, kind=kind)
            assert_array_equal(c, ec)
            a[0], a[3] = (4, 8)
            ec = np.array([True, False, True, False])
            c = in1d(a, b, assume_unique=True, kind=kind)
            assert_array_equal(c, ec)
            a = np.array([5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5])
            b = [2, 3, 4] * mult
            ec = [False, True, False, True, True, True, True, True, True, False, True, False, False, False]
            c = in1d(a, b, kind=kind)
            assert_array_equal(c, ec)
            b = b + [5, 5, 4] * mult
            ec = [True, True, True, True, True, True, True, True, True, True, True, False, True, True]
            c = in1d(a, b, kind=kind)
            assert_array_equal(c, ec)
            a = np.array([5, 7, 1, 2])
            b = np.array([2, 4, 3, 1, 5] * mult)
            ec = np.array([True, False, True, True])
            c = in1d(a, b, kind=kind)
            assert_array_equal(c, ec)
            a = np.array([5, 7, 1, 1, 2])
            b = np.array([2, 4, 3, 3, 1, 5] * mult)
            ec = np.array([True, False, True, True, True])
            c = in1d(a, b, kind=kind)
            assert_array_equal(c, ec)
            a = np.array([5, 5])
            b = np.array([2, 2] * mult)
            ec = np.array([False, False])
            c = in1d(a, b, kind=kind)
            assert_array_equal(c, ec)
        a = np.array([5])
        b = np.array([2])
        ec = np.array([False])
        c = in1d(a, b, kind=kind)
        assert_array_equal(c, ec)
        if kind in {None, 'sort'}:
            assert_array_equal(in1d([], [], kind=kind), [])

    def test_in1d_char_array(self):
        a = np.array(['a', 'b', 'c', 'd', 'e', 'c', 'e', 'b'])
        b = np.array(['a', 'c'])
        ec = np.array([True, False, True, False, False, True, False, False])
        c = in1d(a, b)
        assert_array_equal(c, ec)

    @pytest.mark.parametrize('kind', [None, 'sort', 'table'])
    def test_in1d_invert(self, kind):
        """Test in1d's invert parameter"""
        for mult in (1, 10):
            a = np.array([5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5])
            b = [2, 3, 4] * mult
            assert_array_equal(np.invert(in1d(a, b, kind=kind)), in1d(a, b, invert=True, kind=kind))
        if kind in {None, 'sort'}:
            for mult in (1, 10):
                a = np.array([5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5], dtype=np.float32)
                b = [2, 3, 4] * mult
                b = np.array(b, dtype=np.float32)
                assert_array_equal(np.invert(in1d(a, b, kind=kind)), in1d(a, b, invert=True, kind=kind))

    @pytest.mark.parametrize('kind', [None, 'sort', 'table'])
    def test_in1d_ravel(self, kind):
        a = np.arange(6).reshape(2, 3)
        b = np.arange(3, 9).reshape(3, 2)
        long_b = np.arange(3, 63).reshape(30, 2)
        ec = np.array([False, False, False, True, True, True])
        assert_array_equal(in1d(a, b, assume_unique=True, kind=kind), ec)
        assert_array_equal(in1d(a, b, assume_unique=False, kind=kind), ec)
        assert_array_equal(in1d(a, long_b, assume_unique=True, kind=kind), ec)
        assert_array_equal(in1d(a, long_b, assume_unique=False, kind=kind), ec)

    def test_in1d_hit_alternate_algorithm(self):
        """Hit the standard isin code with integers"""
        a = np.array([5, 4, 5, 3, 4, 4, 1000000000.0], dtype=np.int64)
        b = np.array([2, 3, 4, 1000000000.0], dtype=np.int64)
        expected = np.array([0, 1, 0, 1, 1, 1, 1], dtype=bool)
        assert_array_equal(expected, in1d(a, b))
        assert_array_equal(np.invert(expected), in1d(a, b, invert=True))
        a = np.array([5, 7, 1, 2], dtype=np.int64)
        b = np.array([2, 4, 3, 1, 5, 1000000000.0], dtype=np.int64)
        ec = np.array([True, False, True, True])
        c = in1d(a, b, assume_unique=True)
        assert_array_equal(c, ec)

    @pytest.mark.parametrize('kind', [None, 'sort', 'table'])
    def test_in1d_boolean(self, kind):
        """Test that in1d works for boolean input"""
        a = np.array([True, False])
        b = np.array([False, False, False])
        expected = np.array([False, True])
        assert_array_equal(expected, in1d(a, b, kind=kind))
        assert_array_equal(np.invert(expected), in1d(a, b, invert=True, kind=kind))

    @pytest.mark.parametrize('kind', [None, 'sort'])
    def test_in1d_timedelta(self, kind):
        """Test that in1d works for timedelta input"""
        rstate = np.random.RandomState(0)
        a = rstate.randint(0, 100, size=10)
        b = rstate.randint(0, 100, size=10)
        truth = in1d(a, b)
        a_timedelta = a.astype('timedelta64[s]')
        b_timedelta = b.astype('timedelta64[s]')
        assert_array_equal(truth, in1d(a_timedelta, b_timedelta, kind=kind))

    def test_in1d_table_timedelta_fails(self):
        a = np.array([0, 1, 2], dtype='timedelta64[s]')
        b = a
        with pytest.raises(ValueError):
            in1d(a, b, kind='table')

    @pytest.mark.parametrize('dtype1,dtype2', [(np.int8, np.int16), (np.int16, np.int8), (np.uint8, np.uint16), (np.uint16, np.uint8), (np.uint8, np.int16), (np.int16, np.uint8)])
    @pytest.mark.parametrize('kind', [None, 'sort', 'table'])
    def test_in1d_mixed_dtype(self, dtype1, dtype2, kind):
        """Test that in1d works as expected for mixed dtype input."""
        is_dtype2_signed = np.issubdtype(dtype2, np.signedinteger)
        ar1 = np.array([0, 0, 1, 1], dtype=dtype1)
        if is_dtype2_signed:
            ar2 = np.array([-128, 0, 127], dtype=dtype2)
        else:
            ar2 = np.array([127, 0, 255], dtype=dtype2)
        expected = np.array([True, True, False, False])
        expect_failure = kind == 'table' and any((dtype1 == np.int8 and dtype2 == np.int16, dtype1 == np.int16 and dtype2 == np.int8))
        if expect_failure:
            with pytest.raises(RuntimeError, match='exceed the maximum'):
                in1d(ar1, ar2, kind=kind)
        else:
            assert_array_equal(in1d(ar1, ar2, kind=kind), expected)

    @pytest.mark.parametrize('kind', [None, 'sort', 'table'])
    def test_in1d_mixed_boolean(self, kind):
        """Test that in1d works as expected for bool/int input."""
        for dtype in np.typecodes['AllInteger']:
            a = np.array([True, False, False], dtype=bool)
            b = np.array([0, 0, 0, 0], dtype=dtype)
            expected = np.array([False, True, True], dtype=bool)
            assert_array_equal(in1d(a, b, kind=kind), expected)
            a, b = (b, a)
            expected = np.array([True, True, True, True], dtype=bool)
            assert_array_equal(in1d(a, b, kind=kind), expected)

    def test_in1d_first_array_is_object(self):
        ar1 = [None]
        ar2 = np.array([1] * 10)
        expected = np.array([False])
        result = np.in1d(ar1, ar2)
        assert_array_equal(result, expected)

    def test_in1d_second_array_is_object(self):
        ar1 = 1
        ar2 = np.array([None] * 10)
        expected = np.array([False])
        result = np.in1d(ar1, ar2)
        assert_array_equal(result, expected)

    def test_in1d_both_arrays_are_object(self):
        ar1 = [None]
        ar2 = np.array([None] * 10)
        expected = np.array([True])
        result = np.in1d(ar1, ar2)
        assert_array_equal(result, expected)

    def test_in1d_both_arrays_have_structured_dtype(self):
        dt = np.dtype([('field1', int), ('field2', object)])
        ar1 = np.array([(1, None)], dtype=dt)
        ar2 = np.array([(1, None)] * 10, dtype=dt)
        expected = np.array([True])
        result = np.in1d(ar1, ar2)
        assert_array_equal(result, expected)

    def test_in1d_with_arrays_containing_tuples(self):
        ar1 = np.array([(1,), 2], dtype=object)
        ar2 = np.array([(1,), 2], dtype=object)
        expected = np.array([True, True])
        result = np.in1d(ar1, ar2)
        assert_array_equal(result, expected)
        result = np.in1d(ar1, ar2, invert=True)
        assert_array_equal(result, np.invert(expected))
        ar1 = np.array([(1,), (2, 1), 1], dtype=object)
        ar1 = ar1[:-1]
        ar2 = np.array([(1,), (2, 1), 1], dtype=object)
        ar2 = ar2[:-1]
        expected = np.array([True, True])
        result = np.in1d(ar1, ar2)
        assert_array_equal(result, expected)
        result = np.in1d(ar1, ar2, invert=True)
        assert_array_equal(result, np.invert(expected))
        ar1 = np.array([(1,), (2, 3), 1], dtype=object)
        ar1 = ar1[:-1]
        ar2 = np.array([(1,), 2], dtype=object)
        expected = np.array([True, False])
        result = np.in1d(ar1, ar2)
        assert_array_equal(result, expected)
        result = np.in1d(ar1, ar2, invert=True)
        assert_array_equal(result, np.invert(expected))

    def test_in1d_errors(self):
        """Test that in1d raises expected errors."""
        ar1 = np.array([1, 2, 3, 4, 5])
        ar2 = np.array([2, 4, 6, 8, 10])
        assert_raises(ValueError, in1d, ar1, ar2, kind='quicksort')
        obj_ar1 = np.array([1, 'a', 3, 'b', 5], dtype=object)
        obj_ar2 = np.array([1, 'a', 3, 'b', 5], dtype=object)
        assert_raises(ValueError, in1d, obj_ar1, obj_ar2, kind='table')
        for dtype in [np.int32, np.int64]:
            ar1 = np.array([-1, 2, 3, 4, 5], dtype=dtype)
            overflow_ar2 = np.array([-1, np.iinfo(dtype).max], dtype=dtype)
            assert_raises(RuntimeError, in1d, ar1, overflow_ar2, kind='table')
            result = np.in1d(ar1, overflow_ar2, kind=None)
            assert_array_equal(result, [True] + [False] * 4)
            result = np.in1d(ar1, overflow_ar2, kind='sort')
            assert_array_equal(result, [True] + [False] * 4)

    def test_union1d(self):
        a = np.array([5, 4, 7, 1, 2])
        b = np.array([2, 4, 3, 3, 2, 1, 5])
        ec = np.array([1, 2, 3, 4, 5, 7])
        c = union1d(a, b)
        assert_array_equal(c, ec)
        x = np.array([[0, 1, 2], [3, 4, 5]])
        y = np.array([0, 1, 2, 3, 4])
        ez = np.array([0, 1, 2, 3, 4, 5])
        z = union1d(x, y)
        assert_array_equal(z, ez)
        assert_array_equal([], union1d([], []))

    def test_setdiff1d(self):
        a = np.array([6, 5, 4, 7, 1, 2, 7, 4])
        b = np.array([2, 4, 3, 3, 2, 1, 5])
        ec = np.array([6, 7])
        c = setdiff1d(a, b)
        assert_array_equal(c, ec)
        a = np.arange(21)
        b = np.arange(19)
        ec = np.array([19, 20])
        c = setdiff1d(a, b)
        assert_array_equal(c, ec)
        assert_array_equal([], setdiff1d([], []))
        a = np.array((), np.uint32)
        assert_equal(setdiff1d(a, []).dtype, np.uint32)

    def test_setdiff1d_unique(self):
        a = np.array([3, 2, 1])
        b = np.array([7, 5, 2])
        expected = np.array([3, 1])
        actual = setdiff1d(a, b, assume_unique=True)
        assert_equal(actual, expected)

    def test_setdiff1d_char_array(self):
        a = np.array(['a', 'b', 'c'])
        b = np.array(['a', 'b', 's'])
        assert_array_equal(setdiff1d(a, b), np.array(['c']))

    def test_manyways(self):
        a = np.array([5, 7, 1, 2, 8])
        b = np.array([9, 8, 2, 4, 3, 1, 5])
        c1 = setxor1d(a, b)
        aux1 = intersect1d(a, b)
        aux2 = union1d(a, b)
        c2 = setdiff1d(aux2, aux1)
        assert_array_equal(c1, c2)