import unittest
from traits.api import (
from traits.testing.optional_dependencies import numpy, requires_numpy
def BaseRangeCompound(*args, **kwargs):
    """
    Compound trait including a BaseRange.
    """
    return Either(impossible, BaseRange(*args, **kwargs))