import unittest
from traits.api import Any, HasStrictTraits, Str
def _the_trait_changed(self, new):
    if self.test is not None:
        self.test.change_events.append(new)