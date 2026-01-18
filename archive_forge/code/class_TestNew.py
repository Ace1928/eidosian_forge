import abc
import unittest
import warnings
from traits.api import ABCHasTraits, ABCMetaHasTraits, HasTraits, Int, Float
class TestNew(unittest.TestCase):
    """ Test that __new__ works correctly.
    """

    def setUp(self):
        self.old_filters = warnings.filters[:]
        warnings.simplefilter('error', DeprecationWarning)

    def tearDown(self):
        warnings.filters[:] = self.old_filters

    def test_new(self):

        class A(HasTraits):
            pass
        A(x=10)