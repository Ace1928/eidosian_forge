import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
class HasLengthConstrainedLists(HasTraits):
    """
    Test class for testing list length validation.
    """
    at_least_two = List(Int, [3, 4], minlen=2)
    at_most_five = List(Int, maxlen=5)
    unconstrained = List(Int)