import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, List
class StaticNotifiers2(HasTraits):
    ok = Float

    def _ok_changed(self, new):
        if not hasattr(self, 'calls'):
            self.calls = []
        self.calls.append(new)
    fail = Float

    def _fail_changed(self, new):
        raise Exception('error')