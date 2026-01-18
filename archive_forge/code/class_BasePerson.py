import pickle
import unittest
from traits.api import HasTraits, Int, PrefixMap, TraitError
class BasePerson(HasTraits):
    married = PrefixMap({'yes': 1, 'yeah': 1, 'no': 0, 'nah': 0})
    default_calls = Int(0)

    def _married_default(self):
        self.default_calls += 1
        return 'nah'