import enum
import unittest
from traits.api import (
from traits.etsconfig.api import ETSConfig
from traits.testing.optional_dependencies import requires_traitsui
class EnumCollectionExample(HasTraits):
    rgb = Enum('red', CustomCollection('red', 'green', 'blue'))
    rgb_char = Enum('r', 'g', 'b')
    numbers = Enum(CustomCollection('one', 'two', 'three'))
    letters = Enum('abcdefg')
    int_set_enum = Enum(1, {1, 2})
    correct_int_set_enum = Enum([1, {1, 2}])
    yes_no = Enum('yes', 'no')
    digits = Enum(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    two_digits = Enum(1, 2)
    single_digit = Enum(8)
    slow_enum = BaseEnum('yes', 'no', 'maybe')