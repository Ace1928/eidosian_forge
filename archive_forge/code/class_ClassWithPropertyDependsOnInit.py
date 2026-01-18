import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
class ClassWithPropertyDependsOnInit(ClassWithInstanceDefaultInit):
    extended_age = Property(depends_on='sample_info.age')

    def _get_extended_age(self):
        return self.sample_info.age