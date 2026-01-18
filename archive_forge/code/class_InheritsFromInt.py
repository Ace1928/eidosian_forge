import unittest
from traits.api import (
from traits.testing.optional_dependencies import numpy, requires_numpy
class InheritsFromInt(int):
    """
    Object that's integer-like by virtue of inheriting from int.
    """
    pass