import numpy as np
from collections.abc import MutableMapping
from .common import TestCase, ut
import h5py
from h5py import File
from h5py import h5a,  h5t
from h5py import AttributeManager
class TestDatatype(BaseAttrs):

    def test_datatype(self):
        self.f['foo'] = np.dtype('f')
        dt = self.f['foo']
        self.assertEqual(list(dt.attrs.keys()), [])
        dt.attrs.create('a', 4.0)
        self.assertEqual(list(dt.attrs.keys()), ['a'])
        self.assertEqual(list(dt.attrs.values()), [4.0])