import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class RaisesArgumentlessRuntimeError(HasTraits):
    x = Int(0)

    def _x_changed(self):
        raise RuntimeError