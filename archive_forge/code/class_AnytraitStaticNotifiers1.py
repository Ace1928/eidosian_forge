import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, Undefined
class AnytraitStaticNotifiers1(HasTraits):
    ok = Float

    def _anytrait_changed(self):
        if not hasattr(self, 'anycalls'):
            self.anycalls = []
        self.anycalls.append(True)