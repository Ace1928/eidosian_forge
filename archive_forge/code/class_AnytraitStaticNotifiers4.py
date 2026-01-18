import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, Undefined
class AnytraitStaticNotifiers4(HasTraits):
    ok = Float

    def _anytrait_changed(self, name, old, new):
        if not hasattr(self, 'anycalls'):
            self.anycalls = []
        self.anycalls.append((name, old, new))