import unittest
from traits.api import (
from traits.testing.optional_dependencies import numpy, requires_numpy
def RangeCompound(*args, **kwargs):
    """
    Compound trait including a Range.
    """
    return Either(impossible, Range(*args, **kwargs))