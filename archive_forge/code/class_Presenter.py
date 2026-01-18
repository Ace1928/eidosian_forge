import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
class Presenter(HasTraits):
    obj = Instance(Dummy)
    y = Property(Int(), depends_on='obj.x')

    def _get_y(self):
        return self.obj.x