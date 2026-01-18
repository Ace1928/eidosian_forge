import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, List
def __ok_changed(self, name, old, new):
    self.calls.append((name, old, new))