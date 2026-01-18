import pickle
import unittest
from unittest import mock
from traits.api import HasTraits, Set, Str
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_set_object import TraitSet, TraitSetEvent
from traits.trait_types import _validate_int
class TestTraitSetEvent(unittest.TestCase):

    def test_trait_set_event_str_representation(self):
        """ Test string representation of the TraitSetEvent class. """
        desired_repr = 'TraitSetEvent(removed=set(), added=set())'
        trait_set_event = TraitSetEvent()
        self.assertEqual(desired_repr, str(trait_set_event))
        self.assertEqual(desired_repr, repr(trait_set_event))

    def test_trait_set_event_subclass_str_representation(self):
        """ Test string representation of a subclass of the TraitSetEvent
        class. """

        class DifferentName(TraitSetEvent):
            pass
        desired_repr = 'DifferentName(removed=set(), added=set())'
        different_name_subclass = DifferentName()
        self.assertEqual(desired_repr, str(different_name_subclass))
        self.assertEqual(desired_repr, repr(different_name_subclass))