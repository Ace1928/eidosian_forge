import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
class ClassWithPropertyObservesDecorated(PersonInfo):
    """ Dummy class to test property with observers setup at init time."""
    discounted = Property(Bool(), observe='age')
    discounted_n_calculations = Int()

    def _get_discounted(self):
        self.discounted_n_calculations += 1
        return self.age > 60
    discounted_events = List()

    @observe('discounted')
    def discounted_updated(self, event):
        self.discounted_events.append(event)