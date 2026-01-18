import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, on_trait_change
@on_trait_change('baz')
def _on_baz_change_notification(self):
    self.bar += 1