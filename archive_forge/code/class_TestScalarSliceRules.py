import numpy as np
import h5py
import h5py._hl.selections as sel
import h5py._hl.selections2 as sel2
from .common import TestCase, ut
class TestScalarSliceRules(BaseSelection):
    """
        Internal feature: selections rules for scalar datasets
    """

    def test_args(self):
        """ Permissible arguments for scalar slicing """
        shape, selection = sel2.read_selections_scalar(self.dsid, ())
        self.assertEqual(shape, None)
        self.assertEqual(selection.get_select_npoints(), 1)
        shape, selection = sel2.read_selections_scalar(self.dsid, (Ellipsis,))
        self.assertEqual(shape, ())
        self.assertEqual(selection.get_select_npoints(), 1)
        with self.assertRaises(ValueError):
            shape, selection = sel2.read_selections_scalar(self.dsid, (1,))
        dsid = self.f.create_dataset('y', (1,)).id
        with self.assertRaises(RuntimeError):
            shape, selection = sel2.read_selections_scalar(dsid, (1,))