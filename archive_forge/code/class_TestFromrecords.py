import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
class TestFromrecords:

    def test_fromrecords(self):
        r = np.rec.fromrecords([[456, 'dbe', 1.2], [2, 'de', 1.3]], names='col1,col2,col3')
        assert_equal(r[0].item(), (456, 'dbe', 1.2))
        assert_equal(r['col1'].dtype.kind, 'i')
        assert_equal(r['col2'].dtype.kind, 'U')
        assert_equal(r['col2'].dtype.itemsize, 12)
        assert_equal(r['col3'].dtype.kind, 'f')

    def test_fromrecords_0len(self):
        """ Verify fromrecords works with a 0-length input """
        dtype = [('a', float), ('b', float)]
        r = np.rec.fromrecords([], dtype=dtype)
        assert_equal(r.shape, (0,))

    def test_fromrecords_2d(self):
        data = [[(1, 2), (3, 4), (5, 6)], [(6, 5), (4, 3), (2, 1)]]
        expected_a = [[1, 3, 5], [6, 4, 2]]
        expected_b = [[2, 4, 6], [5, 3, 1]]
        r1 = np.rec.fromrecords(data, dtype=[('a', int), ('b', int)])
        assert_equal(r1['a'], expected_a)
        assert_equal(r1['b'], expected_b)
        r2 = np.rec.fromrecords(data, names=['a', 'b'])
        assert_equal(r2['a'], expected_a)
        assert_equal(r2['b'], expected_b)
        assert_equal(r1, r2)

    def test_method_array(self):
        r = np.rec.array(b'abcdefg' * 100, formats='i2,a3,i4', shape=3, byteorder='big')
        assert_equal(r[1].item(), (25444, b'efg', 1633837924))

    def test_method_array2(self):
        r = np.rec.array([(1, 11, 'a'), (2, 22, 'b'), (3, 33, 'c'), (4, 44, 'd'), (5, 55, 'ex'), (6, 66, 'f'), (7, 77, 'g')], formats='u1,f4,a1')
        assert_equal(r[1].item(), (2, 22.0, b'b'))

    def test_recarray_slices(self):
        r = np.rec.array([(1, 11, 'a'), (2, 22, 'b'), (3, 33, 'c'), (4, 44, 'd'), (5, 55, 'ex'), (6, 66, 'f'), (7, 77, 'g')], formats='u1,f4,a1')
        assert_equal(r[1::2][1].item(), (4, 44.0, b'd'))

    def test_recarray_fromarrays(self):
        x1 = np.array([1, 2, 3, 4])
        x2 = np.array(['a', 'dd', 'xyz', '12'])
        x3 = np.array([1.1, 2, 3, 4])
        r = np.rec.fromarrays([x1, x2, x3], names='a,b,c')
        assert_equal(r[1].item(), (2, 'dd', 2.0))
        x1[1] = 34
        assert_equal(r.a, np.array([1, 2, 3, 4]))

    def test_recarray_fromfile(self):
        data_dir = path.join(path.dirname(__file__), 'data')
        filename = path.join(data_dir, 'recarray_from_file.fits')
        fd = open(filename, 'rb')
        fd.seek(2880 * 2)
        r1 = np.rec.fromfile(fd, formats='f8,i4,a5', shape=3, byteorder='big')
        fd.seek(2880 * 2)
        r2 = np.rec.array(fd, formats='f8,i4,a5', shape=3, byteorder='big')
        fd.seek(2880 * 2)
        bytes_array = BytesIO()
        bytes_array.write(fd.read())
        bytes_array.seek(0)
        r3 = np.rec.fromfile(bytes_array, formats='f8,i4,a5', shape=3, byteorder='big')
        fd.close()
        assert_equal(r1, r2)
        assert_equal(r2, r3)

    def test_recarray_from_obj(self):
        count = 10
        a = np.zeros(count, dtype='O')
        b = np.zeros(count, dtype='f8')
        c = np.zeros(count, dtype='f8')
        for i in range(len(a)):
            a[i] = list(range(1, 10))
        mine = np.rec.fromarrays([a, b, c], names='date,data1,data2')
        for i in range(len(a)):
            assert_(mine.date[i] == list(range(1, 10)))
            assert_(mine.data1[i] == 0.0)
            assert_(mine.data2[i] == 0.0)

    def test_recarray_repr(self):
        a = np.array([(1, 0.1), (2, 0.2)], dtype=[('foo', '<i4'), ('bar', '<f8')])
        a = np.rec.array(a)
        assert_equal(repr(a), textwrap.dedent("            rec.array([(1, 0.1), (2, 0.2)],\n                      dtype=[('foo', '<i4'), ('bar', '<f8')])"))
        a = np.array(np.ones(4, dtype='f8'))
        assert_(repr(np.rec.array(a)).startswith('rec.array'))
        a = np.rec.array(np.ones(3, dtype='i4,i4'))
        assert_equal(repr(a).find('numpy.record'), -1)
        a = np.rec.array(np.ones(3, dtype='i4'))
        assert_(repr(a).find('dtype=int32') != -1)

    def test_0d_recarray_repr(self):
        arr_0d = np.rec.array((1, 2.0, '2003'), dtype='<i4,<f8,<M8[Y]')
        assert_equal(repr(arr_0d), textwrap.dedent("            rec.array((1, 2., '2003'),\n                      dtype=[('f0', '<i4'), ('f1', '<f8'), ('f2', '<M8[Y]')])"))
        record = arr_0d[()]
        assert_equal(repr(record), "(1, 2., '2003')")
        try:
            np.set_printoptions(legacy='1.13')
            assert_equal(repr(record), '(1, 2.0, datetime.date(2003, 1, 1))')
        finally:
            np.set_printoptions(legacy=False)

    def test_recarray_from_repr(self):
        a = np.array([(1, 'ABC'), (2, 'DEF')], dtype=[('foo', int), ('bar', 'S4')])
        recordarr = np.rec.array(a)
        recarr = a.view(np.recarray)
        recordview = a.view(np.dtype((np.record, a.dtype)))
        recordarr_r = eval('numpy.' + repr(recordarr), {'numpy': np})
        recarr_r = eval('numpy.' + repr(recarr), {'numpy': np})
        recordview_r = eval('numpy.' + repr(recordview), {'numpy': np})
        assert_equal(type(recordarr_r), np.recarray)
        assert_equal(recordarr_r.dtype.type, np.record)
        assert_equal(recordarr, recordarr_r)
        assert_equal(type(recarr_r), np.recarray)
        assert_equal(recarr_r.dtype.type, np.record)
        assert_equal(recarr, recarr_r)
        assert_equal(type(recordview_r), np.ndarray)
        assert_equal(recordview.dtype.type, np.record)
        assert_equal(recordview, recordview_r)

    def test_recarray_views(self):
        a = np.array([(1, 'ABC'), (2, 'DEF')], dtype=[('foo', int), ('bar', 'S4')])
        b = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        assert_equal(np.rec.array(a).dtype.type, np.record)
        assert_equal(type(np.rec.array(a)), np.recarray)
        assert_equal(np.rec.array(b).dtype.type, np.int64)
        assert_equal(type(np.rec.array(b)), np.recarray)
        assert_equal(a.view(np.recarray).dtype.type, np.record)
        assert_equal(type(a.view(np.recarray)), np.recarray)
        assert_equal(b.view(np.recarray).dtype.type, np.int64)
        assert_equal(type(b.view(np.recarray)), np.recarray)
        r = np.rec.array(np.ones(4, dtype='f4,i4'))
        rv = r.view('f8').view('f4,i4')
        assert_equal(type(rv), np.recarray)
        assert_equal(rv.dtype.type, np.record)
        r = np.rec.array(np.ones(4, dtype=[('a', 'i4'), ('b', 'i4'), ('c', 'i4,i4')]))
        assert_equal(r['c'].dtype.type, np.record)
        assert_equal(type(r['c']), np.recarray)

        class C(np.recarray):
            pass
        c = r.view(C)
        assert_equal(type(c['c']), C)
        test_dtype = [('a', 'f4,f4'), ('b', 'V8'), ('c', ('f4', 2)), ('d', ('i8', 'i4,i4'))]
        r = np.rec.array([((1, 1), b'11111111', [1, 1], 1), ((1, 1), b'11111111', [1, 1], 1)], dtype=test_dtype)
        assert_equal(r.a.dtype.type, np.record)
        assert_equal(r.b.dtype.type, np.void)
        assert_equal(r.c.dtype.type, np.float32)
        assert_equal(r.d.dtype.type, np.int64)
        r = np.rec.array(np.ones(4, dtype='i4,i4'))
        assert_equal(r.view('f4,f4').dtype.type, np.record)
        assert_equal(r.view(('i4', 2)).dtype.type, np.int32)
        assert_equal(r.view('V8').dtype.type, np.void)
        assert_equal(r.view(('i8', 'i4,i4')).dtype.type, np.int64)
        arrs = [np.ones(4, dtype='f4,i4'), np.ones(4, dtype='f8')]
        for arr in arrs:
            rec = np.rec.array(arr)
            arr2 = rec.view(rec.dtype.fields or rec.dtype, np.ndarray)
            assert_equal(arr2.dtype.type, arr.dtype.type)
            assert_equal(type(arr2), type(arr))

    def test_recarray_from_names(self):
        ra = np.rec.array([(1, 'abc', 3.700000286102295, 0), (2, 'xy', 6.699999809265137, 1), (0, ' ', 0.4000000059604645, 0)], names='c1, c2, c3, c4')
        pa = np.rec.fromrecords([(1, 'abc', 3.700000286102295, 0), (2, 'xy', 6.699999809265137, 1), (0, ' ', 0.4000000059604645, 0)], names='c1, c2, c3, c4')
        assert_(ra.dtype == pa.dtype)
        assert_(ra.shape == pa.shape)
        for k in range(len(ra)):
            assert_(ra[k].item() == pa[k].item())

    def test_recarray_conflict_fields(self):
        ra = np.rec.array([(1, 'abc', 2.3), (2, 'xyz', 4.2), (3, 'wrs', 1.3)], names='field, shape, mean')
        ra.mean = [1.1, 2.2, 3.3]
        assert_array_almost_equal(ra['mean'], [1.1, 2.2, 3.3])
        assert_(type(ra.mean) is type(ra.var))
        ra.shape = (1, 3)
        assert_(ra.shape == (1, 3))
        ra.shape = ['A', 'B', 'C']
        assert_array_equal(ra['shape'], [['A', 'B', 'C']])
        ra.field = 5
        assert_array_equal(ra['field'], [[5, 5, 5]])
        assert_(isinstance(ra.field, collections.abc.Callable))

    def test_fromrecords_with_explicit_dtype(self):
        a = np.rec.fromrecords([(1, 'a'), (2, 'bbb')], dtype=[('a', int), ('b', object)])
        assert_equal(a.a, [1, 2])
        assert_equal(a[0].a, 1)
        assert_equal(a.b, ['a', 'bbb'])
        assert_equal(a[-1].b, 'bbb')
        ndtype = np.dtype([('a', int), ('b', object)])
        a = np.rec.fromrecords([(1, 'a'), (2, 'bbb')], dtype=ndtype)
        assert_equal(a.a, [1, 2])
        assert_equal(a[0].a, 1)
        assert_equal(a.b, ['a', 'bbb'])
        assert_equal(a[-1].b, 'bbb')

    def test_recarray_stringtypes(self):
        a = np.array([('abc ', 1), ('abc', 2)], dtype=[('foo', 'S4'), ('bar', int)])
        a = a.view(np.recarray)
        assert_equal(a.foo[0] == a.foo[1], False)

    def test_recarray_returntypes(self):
        qux_fields = {'C': (np.dtype('S5'), 0), 'D': (np.dtype('S5'), 6)}
        a = np.rec.array([('abc ', (1, 1), 1, ('abcde', 'fgehi')), ('abc', (2, 3), 1, ('abcde', 'jklmn'))], dtype=[('foo', 'S4'), ('bar', [('A', int), ('B', int)]), ('baz', int), ('qux', qux_fields)])
        assert_equal(type(a.foo), np.ndarray)
        assert_equal(type(a['foo']), np.ndarray)
        assert_equal(type(a.bar), np.recarray)
        assert_equal(type(a['bar']), np.recarray)
        assert_equal(a.bar.dtype.type, np.record)
        assert_equal(type(a['qux']), np.recarray)
        assert_equal(a.qux.dtype.type, np.record)
        assert_equal(dict(a.qux.dtype.fields), qux_fields)
        assert_equal(type(a.baz), np.ndarray)
        assert_equal(type(a['baz']), np.ndarray)
        assert_equal(type(a[0].bar), np.record)
        assert_equal(type(a[0]['bar']), np.record)
        assert_equal(a[0].bar.A, 1)
        assert_equal(a[0].bar['A'], 1)
        assert_equal(a[0]['bar'].A, 1)
        assert_equal(a[0]['bar']['A'], 1)
        assert_equal(a[0].qux.D, b'fgehi')
        assert_equal(a[0].qux['D'], b'fgehi')
        assert_equal(a[0]['qux'].D, b'fgehi')
        assert_equal(a[0]['qux']['D'], b'fgehi')

    def test_zero_width_strings(self):
        cols = [['test'] * 3, [''] * 3]
        rec = np.rec.fromarrays(cols)
        assert_equal(rec['f0'], ['test', 'test', 'test'])
        assert_equal(rec['f1'], ['', '', ''])
        dt = np.dtype([('f0', '|S4'), ('f1', '|S')])
        rec = np.rec.fromarrays(cols, dtype=dt)
        assert_equal(rec.itemsize, 4)
        assert_equal(rec['f0'], [b'test', b'test', b'test'])
        assert_equal(rec['f1'], [b'', b'', b''])