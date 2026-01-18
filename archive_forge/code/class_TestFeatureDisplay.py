from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
class TestFeatureDisplay(TestCase):

    def test_feature(self):
        c = commands.FeatureCommand(b'dwim')
        self.assertEqual(b'feature dwim', bytes(c))

    def test_feature_with_value(self):
        c = commands.FeatureCommand(b'dwim', b'please')
        self.assertEqual(b'feature dwim=please', bytes(c))