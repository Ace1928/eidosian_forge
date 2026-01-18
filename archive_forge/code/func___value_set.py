import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
def __value_set(self, value):
    old_value = self.__dict__.get('_value', 0)
    if value != old_value:
        self._value = value
        self.trait_property_changed('value', old_value, value)