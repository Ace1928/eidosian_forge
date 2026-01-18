import unittest
from traits.api import HasTraits, Int, List, Range, Str, TraitError, Tuple
class HasRangeInTuple(HasTraits):
    low = Int()
    high = Int()
    hours_and_name = Tuple(Range(low='low', high='high'), Str)