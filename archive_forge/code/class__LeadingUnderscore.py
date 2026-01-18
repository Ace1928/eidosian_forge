import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, List
class _LeadingUnderscore(HasTraits):
    _ok = Float()
    calls = List()

    def __ok_changed(self, name, old, new):
        self.calls.append((name, old, new))