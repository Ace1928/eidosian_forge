import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, Undefined
class AnytraitStaticNotifiers0(HasTraits):
    ok = Float

    def _anytrait_changed():
        anycalls_0.append(True)