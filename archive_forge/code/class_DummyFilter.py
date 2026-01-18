import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.trait_base import Undefined, Uninitialized
from traits.trait_types import Float, Instance, Int, List
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._testing import (
class DummyFilter:
    """ A callable to be used as the 'filter' for FilteredTraitObserver
    """

    def __init__(self, return_value):
        self.return_value = return_value

    def __call__(self, name, trait):
        return self.return_value

    def __eq__(self, other):
        return self.return_value == other.return_value

    def __hash__(self):
        return hash(self.return_value)

    def __repr__(self):
        formatted_args = [f'return_value={self.return_value!r}']
        return f'{self.__class__.__name__}({', '.join(formatted_args)})'