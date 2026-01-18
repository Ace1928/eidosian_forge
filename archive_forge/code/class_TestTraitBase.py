import collections
import enum
import unittest
from traits.trait_base import safe_contains
class TestTraitBase(unittest.TestCase):

    def test_safe_contains(self):
        self.assertFalse(safe_contains(1, Lights))
        self.assertFalse(safe_contains(MoreLights.amber, Lights))
        self.assertTrue(safe_contains(Lights.red, Lights))
        lights_list = list(Lights)
        self.assertFalse(safe_contains(1, lights_list))
        self.assertFalse(safe_contains(MoreLights.amber, lights_list))
        self.assertTrue(safe_contains(Lights.red, lights_list))
        unfriendly_container = RaisingContainer()
        self.assertFalse(safe_contains(1, unfriendly_container))
        self.assertTrue(safe_contains(1729, unfriendly_container))
        self.assertFalse(safe_contains(Lights.green, unfriendly_container))