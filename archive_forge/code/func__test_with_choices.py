import re
import unittest
from oslo_config import types
def _test_with_choices(self, t):
    self.assertRaises(ValueError, t, 1)
    self.assertRaises(ValueError, t, 200)
    self.assertRaises(ValueError, t, -457)
    t(80)
    t(457)