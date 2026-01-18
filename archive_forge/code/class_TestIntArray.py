from unittest import TestCase
from traitlets import HasTraits, TraitError, observe, Undefined
from traitlets.tests.test_traitlets import TraitTestBase
from traittypes import Array, DataFrame, Series, Dataset, DataArray
import numpy as np
import pandas as pd
import xarray as xr
class TestIntArray(TraitTestBase):
    """
    Test dtype validation with a ``dtype=np.int``
    """
    obj = IntArrayTrait()
    _good_values = [1, [1, 2, 3], [[1, 2, 3], [4, 5, 6]], np.array([1])]
    _bad_values = [[1, [0, 0]]]

    def assertEqual(self, v1, v2):
        return np.testing.assert_array_equal(v1, v2)