import unittest
from traits.api import (
def _sum_changed(self, old, new):
    self.calls += 1
    self.tc.assertEqual(old, self.exp_old)
    self.tc.assertEqual(new, self.exp_new)