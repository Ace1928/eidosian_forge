import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, List
class StaticNotifiers3(HasTraits):
    ok = Float

    def _ok_changed(self, old, new):
        if not hasattr(self, 'calls'):
            self.calls = []
        self.calls.append((old, new))
    fail = Float

    def _fail_changed(self, old, new):
        raise Exception('error')