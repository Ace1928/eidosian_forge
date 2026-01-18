import pickle
import unittest
from traits.api import Expression, HasTraits, Int, TraitError
class BaseFoo(HasTraits):
    bar = Expression()
    default_calls = Int(0)

    def _bar_default(self):
        self.default_calls += 1
        return '1'