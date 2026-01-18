import unittest
from traits.api import (
from traits.testing.optional_dependencies import numpy, requires_numpy
class ModelWithRanges(HasTraits):
    """
        Model containing various Range-like traits.
        """
    percentage = RangeFactory(0.0, 100.0)
    open_closed = RangeFactory(0.0, 100.0, exclude_low=True)
    closed_open = RangeFactory(0.0, 100.0, exclude_high=True)
    open = RangeFactory(0.0, 100.0, exclude_low=True, exclude_high=True)
    closed = RangeFactory(0.0, 100.0)
    steam_temperature = RangeFactory(low=100.0)
    ice_temperature = Range(high=0.0)