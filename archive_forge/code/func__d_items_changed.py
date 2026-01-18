import unittest
from traits.api import HasTraits, Dict
def _d_items_changed(self, event):
    if self.callback:
        self.callback(event)