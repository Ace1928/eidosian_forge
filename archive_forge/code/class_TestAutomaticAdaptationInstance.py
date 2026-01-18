import unittest
from traits.adaptation.api import reset_global_adaptation_manager
from traits.api import (
class TestAutomaticAdaptationInstance(TestAutomaticAdaptationBase, unittest.TestCase):
    """
    Tests for automatic adaptation with Instance.
    """
    trait_under_test = Instance