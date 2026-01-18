import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, on_trait_change
@on_trait_change('fuz')
def _on_fuz_change_notification(self):
    self.bar += 1
    raise FuzException('method')