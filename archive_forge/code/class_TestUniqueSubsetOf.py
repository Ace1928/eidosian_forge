import numpy as np
import numpy.testing as npt
import pytest
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
from hypothesis import given
from hypothesis.extra.array_api import make_strategies_namespace
from xarray.core.variable import Variable
from xarray.testing.strategies import (
from xarray.tests import requires_numpy_array_api
class TestUniqueSubsetOf:

    @given(st.data())
    def test_invalid(self, data):
        with pytest.raises(TypeError, match='must be an Iterable or a Mapping'):
            data.draw(unique_subset_of(0))
        with pytest.raises(ValueError, match='length-zero object'):
            data.draw(unique_subset_of({}))

    @given(st.data(), dimension_sizes(min_dims=1))
    def test_mapping(self, data, dim_sizes):
        subset_of_dim_sizes = data.draw(unique_subset_of(dim_sizes))
        for dim, length in subset_of_dim_sizes.items():
            assert dim in dim_sizes
            assert dim_sizes[dim] == length

    @given(st.data(), dimension_names(min_dims=1))
    def test_iterable(self, data, dim_names):
        subset_of_dim_names = data.draw(unique_subset_of(dim_names))
        for dim in subset_of_dim_names:
            assert dim in dim_names