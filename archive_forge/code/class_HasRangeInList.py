import unittest
from traits.api import HasTraits, Int, List, Range, Str, TraitError, Tuple
class HasRangeInList(HasTraits):
    low = Int()
    high = Int()
    digit_sequence = List(Range(low='low', high='high'))