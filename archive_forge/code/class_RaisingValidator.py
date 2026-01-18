import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
class RaisingValidator(TraitType):
    """ Trait type whose ``validate`` method raises an inappropriate exception.

    Used for testing propagation of that exception.
    """
    info_text = 'bogus'
    default_value_type = DefaultValue.constant
    default_value = None

    def validate(self, object, name, value):
        raise ZeroDivisionError('Just testing')