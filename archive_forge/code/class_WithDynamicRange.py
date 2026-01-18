import unittest
from traits.api import HasTraits, Int, List, Range, Str, TraitError, Tuple
class WithDynamicRange(HasTraits):
    low = Int(0)
    high = Int(10)
    value = Int(3)
    r = Range(value='value', low='low', high='high', exclude_high=True)

    def _r_changed(self, old, new):
        self._changed_handler_calls += 1