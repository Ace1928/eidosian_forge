import unittest
from unittest import mock
from traits.api import (
from traits.observation import _has_traits_helpers as helpers
from traits.observation import expression
from traits.observation.observe import observe
class ObjectWithEqualityComparisonMode(HasTraits):
    """ Class for supporting TestHasTraitsHelpersComparisonMode """
    list_values = List(comparison_mode=ComparisonMode.equality)
    dict_values = Dict(comparison_mode=ComparisonMode.equality)
    set_values = Set(comparison_mode=ComparisonMode.equality)
    number = Any(comparison_mode=ComparisonMode.equality)
    calculated = Property(depends_on='number')

    def _get_calculated(self):
        return None