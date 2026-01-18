import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
def _info_with_default_default(self):
    self.info_with_default_computed = True
    return PersonInfo(age=12)