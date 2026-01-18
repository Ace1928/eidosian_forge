import unittest
from traits.api import BaseFloat, Either, Float, HasTraits, Str, TraitError
from traits.testing.optional_dependencies import numpy, requires_numpy
class TestBaseFloat(unittest.TestCase, CommonFloatTests):

    def setUp(self):
        self.test_class = BaseFloatModel