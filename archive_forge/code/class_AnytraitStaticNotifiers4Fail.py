import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, Undefined
class AnytraitStaticNotifiers4Fail(HasTraits):
    fail = Float

    def _anytrait_changed(self, name, old, new):
        raise Exception('error')