import unittest
from traits.api import HasTraits, Int, List
def _l_items_changed(self, event):
    self.l_events.append(event)