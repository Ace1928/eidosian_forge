import unittest
from traits.api import Float, HasTraits, Int, List
def _y_changed(self, new):
    self.y_changes.append(new)