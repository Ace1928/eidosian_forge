import enum
import unittest
from traits.api import (
from traits.etsconfig.api import ETSConfig
from traits.testing.optional_dependencies import requires_traitsui
class HasEnumInTuple(HasTraits):
    months = List(Int, value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    year_and_month = Tuple(Int(), Enum(values='months'))