import sys
import numpy as np
import h5py
from .common import ut, TestCase
class Test1DFloat(TestCase):

    def setUp(self):
        TestCase.setUp(self)
        self.data = np.arange(13).astype('f')
        self.dset = self.f.create_dataset('x', data=self.data)

    def test_ndim(self):
        """ Verify number of dimensions """
        self.assertEqual(self.dset.ndim, 1)

    def test_shape(self):
        """ Verify shape """
        self.assertEqual(self.dset.shape, (13,))

    def test_ellipsis(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[...])

    def test_tuple(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[()])

    def test_slice_simple(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[0:4])

    def test_slice_zerosize(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[4:4])

    def test_slice_strides(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[1:7:3])

    def test_slice_negindexes(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[-8:-2:3])

    def test_slice_stop_less_than_start(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[7:5])

    def test_slice_outofrange(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[100:400:3])

    def test_slice_backwards(self):
        """ we disallow negative steps """
        with self.assertRaises(ValueError):
            self.dset[::-1]

    def test_slice_zerostride(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[::0])

    def test_index_simple(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[3])

    def test_index_neg(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[-4])

    def test_index_none(self):
        with self.assertRaises(TypeError):
            self.dset[None]

    def test_index_illegal(self):
        """ Illegal slicing argument """
        with self.assertRaises(TypeError):
            self.dset[{}]

    def test_index_outofrange(self):
        with self.assertRaises(IndexError):
            self.dset[100]

    def test_indexlist_simple(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[[1, 2, 5]])

    def test_indexlist_numpyarray(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[np.array([1, 2, 5])])

    def test_indexlist_single_index_ellipsis(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[[0], ...])

    def test_indexlist_numpyarray_single_index_ellipsis(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[np.array([0]), ...])

    def test_indexlist_numpyarray_ellipsis(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[np.array([1, 2, 5]), ...])

    def test_indexlist_empty(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[[]])

    def test_indexlist_outofrange(self):
        with self.assertRaises(IndexError):
            self.dset[[100]]

    def test_indexlist_nonmonotonic(self):
        """ we require index list values to be strictly increasing """
        with self.assertRaises(TypeError):
            self.dset[[1, 3, 2]]

    def test_indexlist_monotonic_negative(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[[0, 2, -2]])
        with self.assertRaises(TypeError):
            self.dset[[-2, -3]]

    def test_indexlist_repeated(self):
        """ we forbid repeated index values """
        with self.assertRaises(TypeError):
            self.dset[[1, 1, 2]]

    def test_mask_true(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[self.data > -100], skip_fast_reader=True)

    def test_mask_false(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[self.data > 100], skip_fast_reader=True)

    def test_mask_partial(self):
        self.assertNumpyBehavior(self.dset, self.data, np.s_[self.data > 5], skip_fast_reader=True)

    def test_mask_wrongsize(self):
        """ we require the boolean mask shape to match exactly """
        with self.assertRaises(TypeError):
            self.dset[np.ones((2,), dtype='bool')]

    def test_fieldnames(self):
        """ field name -> ValueError (no fields) """
        with self.assertRaises(ValueError):
            self.dset['field']