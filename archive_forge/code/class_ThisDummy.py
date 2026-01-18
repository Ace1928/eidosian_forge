import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class ThisDummy(HasTraits):
    allows_none = This()
    disallows_none = This(allow_none=False)