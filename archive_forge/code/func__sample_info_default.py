import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
def _sample_info_default(self):
    self.sample_info_default_computed = True
    return PersonInfo(age=self.info_without_default.age)