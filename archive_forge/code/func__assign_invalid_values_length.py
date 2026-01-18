from traits.api import HasTraits, TraitError
from traits.testing.unittest_tools import UnittestTools
def _assign_invalid_values_length(self, values):
    dummy = self._create_class()
    with self.assertRaises(TraitError):
        dummy.t1 = values
    self.assertEqual(dummy.t1, VALUES)
    with self.assertRaises(TraitError):
        dummy.t2 = values
    self.assertEqual(dummy.t2, VALUES)