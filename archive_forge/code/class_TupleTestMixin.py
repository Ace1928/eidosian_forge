from traits.api import HasTraits, TraitError
from traits.testing.unittest_tools import UnittestTools
class TupleTestMixin(UnittestTools):
    """ A mixin class for testing tuple like traits.

    TestCases should set the self.trait attribute during setUp for the tests
    to run.

    """

    def test_default_values(self):
        dummy = self._create_class()
        self.assertEqual(dummy.t1, VALUES)
        self.assertEqual(dummy.t2, VALUES)

    def test_simple_assignment(self):
        dummy = self._create_class()
        with self.assertTraitChanges(dummy, 't1'):
            dummy.t1 = ('other value 1', 77, None)
        with self.assertTraitChanges(dummy, 't2'):
            dummy.t2 = ('other value 2', 99, None)

    def test_invalid_assignment_length(self):
        self._assign_invalid_values_length(('str', 44))
        self._assign_invalid_values_length(('str', 33, None, []))

    def test_type_checking(self):
        dummy = self._create_class()
        other_tuple = ('other value', 75, True)
        with self.assertRaises(TraitError):
            dummy.t1 = other_tuple
        self.assertEqual(dummy.t1, VALUES)
        try:
            dummy.t2 = other_tuple
        except TraitError:
            self.fail('Unexpected TraitError when assigning to tuple.')
        self.assertEqual(dummy.t2, other_tuple)

    def _assign_invalid_values_length(self, values):
        dummy = self._create_class()
        with self.assertRaises(TraitError):
            dummy.t1 = values
        self.assertEqual(dummy.t1, VALUES)
        with self.assertRaises(TraitError):
            dummy.t2 = values
        self.assertEqual(dummy.t2, VALUES)

    def _create_class(self):
        trait = self.trait

        class Dummy(HasTraits):
            t1 = trait(VALUES)
            t2 = trait(*VALUES)
        return Dummy()