import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, on_trait_change
@on_trait_change('bar')
def _on_bar_change_notification(self):
    pass