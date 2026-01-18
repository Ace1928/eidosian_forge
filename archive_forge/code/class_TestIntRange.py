import unittest
from traits.api import (
from traits.testing.optional_dependencies import numpy, requires_numpy
class TestIntRange(CommonRangeTests, unittest.TestCase):

    def setUp(self):
        self.model = ModelWithRange()