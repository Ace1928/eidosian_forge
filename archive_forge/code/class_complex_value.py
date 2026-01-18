import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class complex_value(HasTraits):
    num1 = Trait(1, Range(1, 5), Range(-5, -1))
    num2 = Trait(1, Range(1, 5), TraitPrefixList('one', 'two', 'three', 'four', 'five'))
    num3 = Trait(1, Range(1, 5), TraitPrefixMap({'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}))
    num4 = Trait(1, Trait(1, Tuple, slow), 10)
    num5 = Trait(1, 10, Trait(1, Tuple, slow))