import unittest
from unittest import mock
from traits.api import (
from traits.observation import _has_traits_helpers as helpers
from traits.observation import expression
from traits.observation.observe import observe
def _int_with_default(self):
    self.int_with_default_computed = True
    return 10