import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, List
class StaticNotifiers1(HasTraits):
    ok = Float

    def _ok_changed(self):
        if not hasattr(self, 'calls'):
            self.calls = []
        self.calls.append(True)
    fail = Float

    def _fail_changed(self):
        raise Exception('error')