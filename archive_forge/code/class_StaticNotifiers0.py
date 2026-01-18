import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, List
class StaticNotifiers0(HasTraits):
    ok = Float

    def _ok_changed():
        calls_0.append(True)
    fail = Float

    def _fail_changed():
        raise Exception('error')