import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
from numpy.ma.mrecords import (
from numpy.ma.testutils import (
from numpy.compat import pickle
class TestMRecordsImport:
    _a = ma.array([1, 2, 3], mask=[0, 0, 1], dtype=int)
    _b = ma.array([1.1, 2.2, 3.3], mask=[0, 0, 1], dtype=float)
    _c = ma.array([b'one', b'two', b'three'], mask=[0, 0, 1], dtype='|S8')
    ddtype = [('a', int), ('b', float), ('c', '|S8')]
    mrec = fromarrays([_a, _b, _c], dtype=ddtype, fill_value=(b'99999', b'99999.', b'N/A'))
    nrec = recfromarrays((_a._data, _b._data, _c._data), dtype=ddtype)
    data = (mrec, nrec, ddtype)

    def test_fromarrays(self):
        _a = ma.array([1, 2, 3], mask=[0, 0, 1], dtype=int)
        _b = ma.array([1.1, 2.2, 3.3], mask=[0, 0, 1], dtype=float)
        _c = ma.array(['one', 'two', 'three'], mask=[0, 0, 1], dtype='|S8')
        mrec, nrec, _ = self.data
        for f, l in zip(('a', 'b', 'c'), (_a, _b, _c)):
            assert_equal(getattr(mrec, f)._mask, l._mask)
        _x = ma.array([1, 1.1, 'one'], mask=[1, 0, 0], dtype=object)
        assert_equal_records(fromarrays(_x, dtype=mrec.dtype), mrec[0])

    def test_fromrecords(self):
        mrec, nrec, ddtype = self.data
        palist = [(1, 'abc', 3.700000286102295, 0), (2, 'xy', 6.699999809265137, 1), (0, ' ', 0.4000000059604645, 0)]
        pa = recfromrecords(palist, names='c1, c2, c3, c4')
        mpa = fromrecords(palist, names='c1, c2, c3, c4')
        assert_equal_records(pa, mpa)
        _mrec = fromrecords(nrec)
        assert_equal(_mrec.dtype, mrec.dtype)
        for field in _mrec.dtype.names:
            assert_equal(getattr(_mrec, field), getattr(mrec._data, field))
        _mrec = fromrecords(nrec.tolist(), names='c1,c2,c3')
        assert_equal(_mrec.dtype, [('c1', int), ('c2', float), ('c3', '|S5')])
        for f, n in zip(('c1', 'c2', 'c3'), ('a', 'b', 'c')):
            assert_equal(getattr(_mrec, f), getattr(mrec._data, n))
        _mrec = fromrecords(mrec)
        assert_equal(_mrec.dtype, mrec.dtype)
        assert_equal_records(_mrec._data, mrec.filled())
        assert_equal_records(_mrec._mask, mrec._mask)

    def test_fromrecords_wmask(self):
        mrec, nrec, ddtype = self.data
        _mrec = fromrecords(nrec.tolist(), dtype=ddtype, mask=[0, 1, 0])
        assert_equal_records(_mrec._data, mrec._data)
        assert_equal(_mrec._mask.tolist(), [(0, 0, 0), (1, 1, 1), (0, 0, 0)])
        _mrec = fromrecords(nrec.tolist(), dtype=ddtype, mask=True)
        assert_equal_records(_mrec._data, mrec._data)
        assert_equal(_mrec._mask.tolist(), [(1, 1, 1), (1, 1, 1), (1, 1, 1)])
        _mrec = fromrecords(nrec.tolist(), dtype=ddtype, mask=mrec._mask)
        assert_equal_records(_mrec._data, mrec._data)
        assert_equal(_mrec._mask.tolist(), mrec._mask.tolist())
        _mrec = fromrecords(nrec.tolist(), dtype=ddtype, mask=mrec._mask.tolist())
        assert_equal_records(_mrec._data, mrec._data)
        assert_equal(_mrec._mask.tolist(), mrec._mask.tolist())

    def test_fromtextfile(self):
        fcontent = '#\n\'One (S)\',\'Two (I)\',\'Three (F)\',\'Four (M)\',\'Five (-)\',\'Six (C)\'\n\'strings\',1,1.0,\'mixed column\',,1\n\'with embedded "double quotes"\',2,2.0,1.0,,1\n\'strings\',3,3.0E5,3,,1\n\'strings\',4,-1e-10,,,1\n'
        with temppath() as path:
            with open(path, 'w') as f:
                f.write(fcontent)
            mrectxt = fromtextfile(path, delimiter=',', varnames='ABCDEFG')
        assert_(isinstance(mrectxt, MaskedRecords))
        assert_equal(mrectxt.F, [1, 1, 1, 1])
        assert_equal(mrectxt.E._mask, [1, 1, 1, 1])
        assert_equal(mrectxt.C, [1, 2, 300000.0, -1e-10])

    def test_addfield(self):
        mrec, nrec, ddtype = self.data
        d, m = ([100, 200, 300], [1, 0, 0])
        mrec = addfield(mrec, ma.array(d, mask=m))
        assert_equal(mrec.f3, d)
        assert_equal(mrec.f3._mask, m)