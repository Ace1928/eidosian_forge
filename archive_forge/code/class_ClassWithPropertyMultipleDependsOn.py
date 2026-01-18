import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
class ClassWithPropertyMultipleDependsOn(PersonInfo):
    """ Dummy class using 'depends_on', to be compared with the one above.
    """
    computed_value = Property(depends_on=['age', 'gender'])
    computed_value_n_calculations = Int()

    def _get_computed_value(self):
        self.computed_value_n_calculations += 1
        return len(self.gender) + self.age