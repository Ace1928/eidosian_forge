import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
class ClassWithInstanceDefaultInit(HasTraits):
    info_without_default = Instance(PersonInfo)
    list_of_infos = List(Instance(PersonInfo), comparison_mode=1)
    sample_info = Instance(PersonInfo)
    sample_info_default_computed = Bool()

    def _sample_info_default(self):
        self.sample_info_default_computed = True
        return PersonInfo(age=self.info_without_default.age)
    info_with_default = Instance(PersonInfo)
    info_with_default_computed = Bool()

    def _info_with_default_default(self):
        self.info_with_default_computed = True
        return PersonInfo(age=12)