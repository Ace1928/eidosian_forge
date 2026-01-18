import sys
import operator
import pytest
import ctypes
import gc
import types
from typing import Any
import numpy as np
import numpy.dtypes
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import create_custom_field_dtype
from numpy.testing import (
from numpy.compat import pickle
from itertools import permutations
import random
import hypothesis
from hypothesis.extra import numpy as hynp
class TestRecord:

    def test_equivalent_record(self):
        """Test whether equivalent record dtypes hash the same."""
        a = np.dtype([('yo', int)])
        b = np.dtype([('yo', int)])
        assert_dtype_equal(a, b)

    def test_different_names(self):
        a = np.dtype([('yo', int)])
        b = np.dtype([('ye', int)])
        assert_dtype_not_equal(a, b)

    def test_different_titles(self):
        a = np.dtype({'names': ['r', 'b'], 'formats': ['u1', 'u1'], 'titles': ['Red pixel', 'Blue pixel']})
        b = np.dtype({'names': ['r', 'b'], 'formats': ['u1', 'u1'], 'titles': ['RRed pixel', 'Blue pixel']})
        assert_dtype_not_equal(a, b)

    @pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
    def test_refcount_dictionary_setting(self):
        names = ['name1']
        formats = ['f8']
        titles = ['t1']
        offsets = [0]
        d = dict(names=names, formats=formats, titles=titles, offsets=offsets)
        refcounts = {k: sys.getrefcount(i) for k, i in d.items()}
        np.dtype(d)
        refcounts_new = {k: sys.getrefcount(i) for k, i in d.items()}
        assert refcounts == refcounts_new

    def test_mutate(self):
        a = np.dtype([('yo', int)])
        b = np.dtype([('yo', int)])
        c = np.dtype([('ye', int)])
        assert_dtype_equal(a, b)
        assert_dtype_not_equal(a, c)
        a.names = ['ye']
        assert_dtype_equal(a, c)
        assert_dtype_not_equal(a, b)
        state = b.__reduce__()[2]
        a.__setstate__(state)
        assert_dtype_equal(a, b)
        assert_dtype_not_equal(a, c)

    def test_mutate_error(self):
        a = np.dtype('i,i')
        with pytest.raises(ValueError, match='must replace all names at once'):
            a.names = ['f0']
        with pytest.raises(ValueError, match='.*and not string'):
            a.names = ['f0', b'not a unicode name']

    def test_not_lists(self):
        """Test if an appropriate exception is raised when passing bad values to
        the dtype constructor.
        """
        assert_raises(TypeError, np.dtype, dict(names={'A', 'B'}, formats=['f8', 'i4']))
        assert_raises(TypeError, np.dtype, dict(names=['A', 'B'], formats={'f8', 'i4'}))

    def test_aligned_size(self):
        dt = np.dtype('i4, i1', align=True)
        assert_equal(dt.itemsize, 8)
        dt = np.dtype([('f0', 'i4'), ('f1', 'i1')], align=True)
        assert_equal(dt.itemsize, 8)
        dt = np.dtype({'names': ['f0', 'f1'], 'formats': ['i4', 'u1'], 'offsets': [0, 4]}, align=True)
        assert_equal(dt.itemsize, 8)
        dt = np.dtype({'f0': ('i4', 0), 'f1': ('u1', 4)}, align=True)
        assert_equal(dt.itemsize, 8)
        dt1 = np.dtype([('f0', 'i4'), ('f1', [('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')]), ('f2', 'i1')], align=True)
        assert_equal(dt1.itemsize, 20)
        dt2 = np.dtype({'names': ['f0', 'f1', 'f2'], 'formats': ['i4', [('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')], 'i1'], 'offsets': [0, 4, 16]}, align=True)
        assert_equal(dt2.itemsize, 20)
        dt3 = np.dtype({'f0': ('i4', 0), 'f1': ([('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')], 4), 'f2': ('i1', 16)}, align=True)
        assert_equal(dt3.itemsize, 20)
        assert_equal(dt1, dt2)
        assert_equal(dt2, dt3)
        dt1 = np.dtype([('f0', 'i4'), ('f1', [('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')]), ('f2', 'i1')], align=False)
        assert_equal(dt1.itemsize, 11)
        dt2 = np.dtype({'names': ['f0', 'f1', 'f2'], 'formats': ['i4', [('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')], 'i1'], 'offsets': [0, 4, 10]}, align=False)
        assert_equal(dt2.itemsize, 11)
        dt3 = np.dtype({'f0': ('i4', 0), 'f1': ([('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')], 4), 'f2': ('i1', 10)}, align=False)
        assert_equal(dt3.itemsize, 11)
        assert_equal(dt1, dt2)
        assert_equal(dt2, dt3)
        dt1 = np.dtype([('a', '|i1'), ('b', [('f0', '<i2'), ('f1', '<f4')], 2)], align=True)
        assert_equal(dt1.descr, [('a', '|i1'), ('', '|V3'), ('b', [('f0', '<i2'), ('', '|V2'), ('f1', '<f4')], (2,))])

    def test_union_struct(self):
        dt = np.dtype({'names': ['f0', 'f1', 'f2'], 'formats': ['<u4', '<u2', '<u2'], 'offsets': [0, 0, 2]}, align=True)
        assert_equal(dt.itemsize, 4)
        a = np.array([3], dtype='<u4').view(dt)
        a['f1'] = 10
        a['f2'] = 36
        assert_equal(a['f0'], 10 + 36 * 256 * 256)
        dt = np.dtype({'names': ['f0', 'f1', 'f2'], 'formats': ['<u4', '<u2', '<u2'], 'offsets': [4, 0, 2]}, align=True)
        assert_equal(dt.itemsize, 8)
        dt2 = np.dtype({'names': ['f2', 'f0', 'f1'], 'formats': ['<u4', '<u2', '<u2'], 'offsets': [4, 0, 2]}, align=True)
        vals = [(0, 1, 2), (3, 2 ** 15 - 1, 4)]
        vals2 = [(0, 1, 2), (3, 2 ** 15 - 1, 4)]
        a = np.array(vals, dt)
        b = np.array(vals2, dt2)
        assert_equal(a.astype(dt2), b)
        assert_equal(b.astype(dt), a)
        assert_equal(a.view(dt2), b)
        assert_equal(b.view(dt), a)
        assert_raises(TypeError, np.dtype, {'names': ['f0', 'f1'], 'formats': ['O', 'i1'], 'offsets': [0, 2]})
        assert_raises(TypeError, np.dtype, {'names': ['f0', 'f1'], 'formats': ['i4', 'O'], 'offsets': [0, 3]})
        assert_raises(TypeError, np.dtype, {'names': ['f0', 'f1'], 'formats': [[('a', 'O')], 'i1'], 'offsets': [0, 2]})
        assert_raises(TypeError, np.dtype, {'names': ['f0', 'f1'], 'formats': ['i4', [('a', 'O')]], 'offsets': [0, 3]})
        dt = np.dtype({'names': ['f0', 'f1'], 'formats': ['i1', 'O'], 'offsets': [np.dtype('intp').itemsize, 0]})

    @pytest.mark.parametrize(['obj', 'dtype', 'expected'], [([], '(2)f4,', np.empty((0, 2), dtype='f4')), (3, '(3)f4,', [3, 3, 3]), (np.float64(2), '(2)f4,', [2, 2]), ([((0, 1), (1, 2)), ((2,),)], '(2,2)f4', None), (['1', '2'], '(2)i,', None)])
    def test_subarray_list(self, obj, dtype, expected):
        dtype = np.dtype(dtype)
        res = np.array(obj, dtype=dtype)
        if expected is None:
            expected = np.empty(len(obj), dtype=dtype)
            for i in range(len(expected)):
                expected[i] = obj[i]
        assert_array_equal(res, expected)

    def test_comma_datetime(self):
        dt = np.dtype('M8[D],datetime64[Y],i8')
        assert_equal(dt, np.dtype([('f0', 'M8[D]'), ('f1', 'datetime64[Y]'), ('f2', 'i8')]))

    def test_from_dictproxy(self):
        dt = np.dtype({'names': ['a', 'b'], 'formats': ['i4', 'f4']})
        assert_dtype_equal(dt, np.dtype(dt.fields))
        dt2 = np.dtype((np.void, dt.fields))
        assert_equal(dt2.fields, dt.fields)

    def test_from_dict_with_zero_width_field(self):
        dt = np.dtype([('val1', np.float32, (0,)), ('val2', int)])
        dt2 = np.dtype({'names': ['val1', 'val2'], 'formats': [(np.float32, (0,)), int]})
        assert_dtype_equal(dt, dt2)
        assert_equal(dt.fields['val1'][0].itemsize, 0)
        assert_equal(dt.itemsize, dt.fields['val2'][0].itemsize)

    def test_bool_commastring(self):
        d = np.dtype('?,?,?')
        assert_equal(len(d.names), 3)
        for n in d.names:
            assert_equal(d.fields[n][0], np.dtype('?'))

    def test_nonint_offsets(self):

        def make_dtype(off):
            return np.dtype({'names': ['A'], 'formats': ['i4'], 'offsets': [off]})
        assert_raises(TypeError, make_dtype, 'ASD')
        assert_raises(OverflowError, make_dtype, 2 ** 70)
        assert_raises(TypeError, make_dtype, 2.3)
        assert_raises(ValueError, make_dtype, -10)
        dt = make_dtype(np.uint32(0))
        np.zeros(1, dtype=dt)[0].item()

    def test_fields_by_index(self):
        dt = np.dtype([('a', np.int8), ('b', np.float32, 3)])
        assert_dtype_equal(dt[0], np.dtype(np.int8))
        assert_dtype_equal(dt[1], np.dtype((np.float32, 3)))
        assert_dtype_equal(dt[-1], dt[1])
        assert_dtype_equal(dt[-2], dt[0])
        assert_raises(IndexError, lambda: dt[-3])
        assert_raises(TypeError, operator.getitem, dt, 3.0)
        assert_equal(dt[1], dt[np.int8(1)])

    @pytest.mark.parametrize('align_flag', [False, True])
    def test_multifield_index(self, align_flag):
        dt = np.dtype([(('title', 'col1'), '<U20'), ('A', '<f8'), ('B', '<f8')], align=align_flag)
        dt_sub = dt[['B', 'col1']]
        assert_equal(dt_sub, np.dtype({'names': ['B', 'col1'], 'formats': ['<f8', '<U20'], 'offsets': [88, 0], 'titles': [None, 'title'], 'itemsize': 96}))
        assert_equal(dt_sub.isalignedstruct, align_flag)
        dt_sub = dt[['B']]
        assert_equal(dt_sub, np.dtype({'names': ['B'], 'formats': ['<f8'], 'offsets': [88], 'itemsize': 96}))
        assert_equal(dt_sub.isalignedstruct, align_flag)
        dt_sub = dt[[]]
        assert_equal(dt_sub, np.dtype({'names': [], 'formats': [], 'offsets': [], 'itemsize': 96}))
        assert_equal(dt_sub.isalignedstruct, align_flag)
        assert_raises(TypeError, operator.getitem, dt, ())
        assert_raises(TypeError, operator.getitem, dt, [1, 2, 3])
        assert_raises(TypeError, operator.getitem, dt, ['col1', 2])
        assert_raises(KeyError, operator.getitem, dt, ['fake'])
        assert_raises(KeyError, operator.getitem, dt, ['title'])
        assert_raises(ValueError, operator.getitem, dt, ['col1', 'col1'])

    def test_partial_dict(self):
        assert_raises(ValueError, np.dtype, {'formats': ['i4', 'i4'], 'f0': ('i4', 0), 'f1': ('i4', 4)})

    def test_fieldless_views(self):
        a = np.zeros(2, dtype={'names': [], 'formats': [], 'offsets': [], 'itemsize': 8})
        assert_raises(ValueError, a.view, np.dtype([]))
        d = np.dtype((np.dtype([]), 10))
        assert_equal(d.shape, (10,))
        assert_equal(d.itemsize, 0)
        assert_equal(d.base, np.dtype([]))
        arr = np.fromiter((() for i in range(10)), [])
        assert_equal(arr.dtype, np.dtype([]))
        assert_raises(ValueError, np.frombuffer, b'', dtype=[])
        assert_equal(np.frombuffer(b'', dtype=[], count=2), np.empty(2, dtype=[]))
        assert_raises(ValueError, np.dtype, ([], 'f8'))
        assert_raises(ValueError, np.zeros(1, dtype='i4').view, [])
        assert_equal(np.zeros(2, dtype=[]) == np.zeros(2, dtype=[]), np.ones(2, dtype=bool))
        assert_equal(np.zeros((1, 2), dtype=[]) == a, np.ones((1, 2), dtype=bool))

    def test_nonstructured_with_object(self):
        arr = np.recarray((0,), dtype='O')
        assert arr.dtype.names is None
        assert arr.dtype.hasobject
        del arr