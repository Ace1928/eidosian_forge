import unittest
from unittest import mock
from traits.api import (
from traits.observation import _has_traits_helpers as helpers
from traits.observation import expression
from traits.observation.observe import observe
class CannotCompare:

    def __eq__(self, other):
        raise TypeError('Cannot be compared for equality.')