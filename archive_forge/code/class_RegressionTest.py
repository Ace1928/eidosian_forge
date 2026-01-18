import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class RegressionTest(unittest.TestCase):
    """ Check that fixed bugs stay fixed.
    """

    def test_factory_subclass_no_segfault(self):
        """ Test that we can provide an instance as a default in the definition
        of a subclass.
        """
        obj = ConsumerSubclass()
        obj.x

    def test_trait_compound_instance(self):
        """ Test that a deferred Instance() embedded in a TraitCompound handler
        and then a list will not replace the validate method for the outermost
        trait.
        """
        d = Dummy()
        d.xl = [HasTraits()]
        d.x = 'OK'