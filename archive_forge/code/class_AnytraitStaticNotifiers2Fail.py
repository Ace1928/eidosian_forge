import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, Undefined
class AnytraitStaticNotifiers2Fail(HasTraits):
    fail = Float

    def _anytrait_changed(self, name):
        raise Exception('error')