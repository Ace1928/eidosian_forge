from __future__ import annotations
import warnings
from abc import ABC
from copy import copy, deepcopy
from datetime import datetime, timedelta
from textwrap import dedent
from typing import Generic
import numpy as np
import pandas as pd
import pytest
import pytz
from xarray import DataArray, Dataset, IndexVariable, Variable, set_options
from xarray.core import dtypes, duck_array_ops, indexing
from xarray.core.common import full_like, ones_like, zeros_like
from xarray.core.indexing import (
from xarray.core.types import T_DuckArray
from xarray.core.utils import NDArrayMixin
from xarray.core.variable import as_compatible_data, as_variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
from xarray.tests.test_namedarray import NamedArraySubclassobjects
@pytest.mark.parametrize('Var', [Variable, IndexVariable])
class TestNumpyCoercion:

    def test_from_numpy(self, Var):
        v = Var('x', [1, 2, 3])
        assert_identical(v.as_numpy(), v)
        np.testing.assert_equal(v.to_numpy(), np.array([1, 2, 3]))

    @requires_dask
    def test_from_dask(self, Var):
        v = Var('x', [1, 2, 3])
        v_chunked = v.chunk(1)
        assert_identical(v_chunked.as_numpy(), v.compute())
        np.testing.assert_equal(v.to_numpy(), np.array([1, 2, 3]))

    @requires_pint
    def test_from_pint(self, Var):
        import pint
        arr = np.array([1, 2, 3])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=pint.UnitStrippedWarning)
            v = Var('x', pint.Quantity(arr, units='m'))
        assert_identical(v.as_numpy(), Var('x', arr))
        np.testing.assert_equal(v.to_numpy(), arr)

    @requires_sparse
    def test_from_sparse(self, Var):
        if Var is IndexVariable:
            pytest.skip("Can't have 2D IndexVariables")
        import sparse
        arr = np.diagflat([1, 2, 3])
        sparr = sparse.COO(coords=[[0, 1, 2], [0, 1, 2]], data=[1, 2, 3])
        v = Variable(['x', 'y'], sparr)
        assert_identical(v.as_numpy(), Variable(['x', 'y'], arr))
        np.testing.assert_equal(v.to_numpy(), arr)

    @requires_cupy
    def test_from_cupy(self, Var):
        if Var is IndexVariable:
            pytest.skip('cupy in default indexes is not supported at the moment')
        import cupy as cp
        arr = np.array([1, 2, 3])
        v = Var('x', cp.array(arr))
        assert_identical(v.as_numpy(), Var('x', arr))
        np.testing.assert_equal(v.to_numpy(), arr)

    @requires_dask
    @requires_pint
    def test_from_pint_wrapping_dask(self, Var):
        import dask
        import pint
        arr = np.array([1, 2, 3])
        d = dask.array.from_array(np.array([1, 2, 3]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=pint.UnitStrippedWarning)
            v = Var('x', pint.Quantity(d, units='m'))
        result = v.as_numpy()
        assert_identical(result, Var('x', arr))
        np.testing.assert_equal(v.to_numpy(), arr)