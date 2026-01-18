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
class TestDimensionNamesStrategy:

    @given(dimension_names())
    def test_types(self, dims):
        assert isinstance(dims, list)
        for d in dims:
            assert isinstance(d, str)

    @given(dimension_names())
    def test_unique(self, dims):
        assert len(set(dims)) == len(dims)

    @given(st.data(), st.tuples(st.integers(0, 10), st.integers(0, 10)).map(sorted))
    def test_number_of_dims(self, data, ndims):
        min_dims, max_dims = ndims
        dim_names = data.draw(dimension_names(min_dims=min_dims, max_dims=max_dims))
        assert isinstance(dim_names, list)
        assert min_dims <= len(dim_names) <= max_dims