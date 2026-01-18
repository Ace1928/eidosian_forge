import unittest
from traits.adaptation.api import reset_global_adaptation_manager
from traits.api import (
class TestAutomaticAdaptationBaseInstance(TestAutomaticAdaptationBase, unittest.TestCase):
    """
    Tests for automatic adaptation with BaseInstance.
    """
    trait_under_test = BaseInstance