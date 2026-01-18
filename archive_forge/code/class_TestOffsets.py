from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
class TestOffsets(TestCase):
    """
        Check that compound members with aligned or manual offsets are handled
        correctly.
    """

    def test_compound_vlen(self):
        vidt = h5py.vlen_dtype(np.uint8)
        eidt = h5py.enum_dtype({'OFF': 0, 'ON': 1}, basetype=np.uint8)
        for np_align in (False, True):
            dt = np.dtype([('a', eidt), ('foo', vidt), ('bar', vidt), ('switch', eidt)], align=np_align)
            np_offsets = [dt.fields[i][1] for i in dt.names]
            for logical in (False, True):
                if logical and np_align:
                    self.assertRaises(TypeError, h5py.h5t.py_create, dt, logical=logical)
                else:
                    ht = h5py.h5t.py_create(dt, logical=logical)
                    offsets = [ht.get_member_offset(i) for i in range(ht.get_nmembers())]
                    if np_align:
                        self.assertEqual(np_offsets, offsets)

    def test_aligned_offsets(self):
        dt = np.dtype('i4,i8,i2', align=True)
        ht = h5py.h5t.py_create(dt)
        self.assertEqual(dt.itemsize, ht.get_size())
        self.assertEqual([dt.fields[i][1] for i in dt.names], [ht.get_member_offset(i) for i in range(ht.get_nmembers())])

    def test_aligned_data(self):
        dt = np.dtype('i4,f8,i2', align=True)
        data = np.zeros(10, dtype=dt)
        data['f0'] = np.array(np.random.randint(-100, 100, size=data.size), dtype='i4')
        data['f1'] = np.random.rand(data.size)
        data['f2'] = np.array(np.random.randint(-100, 100, size=data.size), dtype='i2')
        fname = self.mktemp()
        with h5py.File(fname, 'w') as f:
            f['data'] = data
        with h5py.File(fname, 'r') as f:
            self.assertArrayEqual(f['data'], data)

    def test_compound_robustness(self):
        fields = [('f0', np.float64, 25), ('f1', np.uint64, 9), ('f2', np.uint32, 0), ('f3', np.uint16, 5)]
        lastfield = fields[np.argmax([x[2] for x in fields])]
        itemsize = lastfield[2] + np.dtype(lastfield[1]).itemsize + 6
        extract_index = lambda index, sequence: [x[index] for x in sequence]
        dt = np.dtype({'names': extract_index(0, fields), 'formats': extract_index(1, fields), 'offsets': extract_index(2, fields), 'itemsize': itemsize})
        self.assertTrue(dt.itemsize == itemsize)
        data = np.zeros(10, dtype=dt)
        f1 = np.array([1 + i * 4 for i in range(data.shape[0])], dtype=dt.fields['f1'][0])
        f2 = np.array([2 + i * 4 for i in range(data.shape[0])], dtype=dt.fields['f2'][0])
        f3 = np.array([3 + i * 4 for i in range(data.shape[0])], dtype=dt.fields['f3'][0])
        f0c = 3.14
        data['f0'] = f0c
        data['f3'] = f3
        data['f1'] = f1
        data['f2'] = f2
        self.assertTrue(np.all(data['f0'] == f0c))
        self.assertArrayEqual(data['f3'], f3)
        self.assertArrayEqual(data['f1'], f1)
        self.assertArrayEqual(data['f2'], f2)
        fname = self.mktemp()
        with h5py.File(fname, 'w') as fd:
            fd.create_dataset('data', data=data)
        with h5py.File(fname, 'r') as fd:
            readback = fd['data']
            self.assertTrue(readback.dtype == dt)
            self.assertArrayEqual(readback, data)
            self.assertTrue(np.all(readback['f0'] == f0c))
            self.assertArrayEqual(readback['f1'], f1)
            self.assertArrayEqual(readback['f2'], f2)
            self.assertArrayEqual(readback['f3'], f3)

    def test_out_of_order_offsets(self):
        dt = np.dtype({'names': ['f1', 'f2', 'f3'], 'formats': ['<f4', '<i4', '<f8'], 'offsets': [0, 16, 8]})
        data = np.zeros(10, dtype=dt)
        data['f1'] = np.random.rand(data.size)
        data['f2'] = np.random.randint(-10, 11, data.size)
        data['f3'] = np.random.rand(data.size) * -1
        fname = self.mktemp()
        with h5py.File(fname, 'w') as fd:
            fd.create_dataset('data', data=data)
        with h5py.File(fname, 'r') as fd:
            self.assertArrayEqual(fd['data'], data)

    def test_float_round_tripping(self):
        dtypes = set((f for f in np.sctypeDict.values() if np.issubdtype(f, np.floating) or np.issubdtype(f, np.complexfloating)))
        unsupported_types = []
        if platform.machine() in UNSUPPORTED_LONG_DOUBLE:
            for x in UNSUPPORTED_LONG_DOUBLE_TYPES:
                if hasattr(np, x):
                    unsupported_types.append(getattr(np, x))
        dtype_dset_map = {str(j): d for j, d in enumerate(dtypes) if d not in unsupported_types}
        fname = self.mktemp()
        with h5py.File(fname, 'w') as f:
            for n, d in dtype_dset_map.items():
                data = np.zeros(10, dtype=d)
                data[...] = np.arange(10)
                f.create_dataset(n, data=data)
        with h5py.File(fname, 'r') as f:
            for n, d in dtype_dset_map.items():
                ldata = f[n][:]
                self.assertEqual(ldata.dtype, d)