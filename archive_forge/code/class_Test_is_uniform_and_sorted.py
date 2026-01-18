from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
class Test_is_uniform_and_sorted:

    def test_sorted_uniform(self):
        assert utils.is_uniform_spaced(np.arange(5))

    def test_sorted_not_uniform(self):
        assert not utils.is_uniform_spaced([-2, 1, 89])

    def test_not_sorted_uniform(self):
        assert not utils.is_uniform_spaced([1, -1, 3])

    def test_not_sorted_not_uniform(self):
        assert not utils.is_uniform_spaced([4, 1, 89])

    def test_two_numbers(self):
        assert utils.is_uniform_spaced([0, 1.7])

    def test_relative_tolerance(self):
        assert utils.is_uniform_spaced([0, 0.97, 2], rtol=0.1)