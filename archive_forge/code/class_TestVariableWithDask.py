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
@requires_dask
class TestVariableWithDask(VariableSubclassobjects):

    def cls(self, *args, **kwargs) -> Variable:
        return Variable(*args, **kwargs).chunk()

    def test_chunk(self):
        unblocked = Variable(['dim_0', 'dim_1'], np.ones((3, 4)))
        assert unblocked.chunks is None
        blocked = unblocked.chunk()
        assert blocked.chunks == ((3,), (4,))
        first_dask_name = blocked.data.name
        blocked = unblocked.chunk(chunks=((2, 1), (2, 2)))
        assert blocked.chunks == ((2, 1), (2, 2))
        assert blocked.data.name != first_dask_name
        blocked = unblocked.chunk(chunks=(3, 3))
        assert blocked.chunks == ((3,), (3, 1))
        assert blocked.data.name != first_dask_name
        assert unblocked.chunk(2).data.name == unblocked.chunk(2).data.name
        assert blocked.load().chunks is None
        import dask.array as da
        blocked = unblocked.chunk(name='testname_')
        assert isinstance(blocked.data, da.Array)
        assert 'testname_' in blocked.data.name
        blocked = unblocked.chunk(dim_0=3, dim_1=3)
        assert blocked.chunks == ((3,), (3, 1))
        assert blocked.data.name != first_dask_name

    @pytest.mark.xfail
    def test_0d_object_array_with_list(self):
        super().test_0d_object_array_with_list()

    @pytest.mark.xfail
    def test_array_interface(self):
        super().test_array_interface()

    @pytest.mark.xfail
    def test_copy_index(self):
        super().test_copy_index()

    @pytest.mark.xfail
    @pytest.mark.filterwarnings('ignore:elementwise comparison failed.*:FutureWarning')
    def test_eq_all_dtypes(self):
        super().test_eq_all_dtypes()

    def test_getitem_fancy(self):
        super().test_getitem_fancy()

    def test_getitem_1d_fancy(self):
        super().test_getitem_1d_fancy()

    def test_getitem_with_mask_nd_indexer(self):
        import dask.array as da
        v = Variable(['x'], da.arange(3, chunks=3))
        indexer = Variable(('x', 'y'), [[0, -1], [-1, 2]])
        assert_identical(v._getitem_with_mask(indexer, fill_value=-1), self.cls(('x', 'y'), [[0, -1], [-1, 2]]))

    @pytest.mark.parametrize('dim', ['x', 'y'])
    @pytest.mark.parametrize('window', [3, 8, 11])
    @pytest.mark.parametrize('center', [True, False])
    def test_dask_rolling(self, dim, window, center):
        import dask
        import dask.array as da
        dask.config.set(scheduler='single-threaded')
        x = Variable(('x', 'y'), np.array(np.random.randn(100, 40), dtype=float))
        dx = Variable(('x', 'y'), da.from_array(x, chunks=[(6, 30, 30, 20, 14), 8]))
        expected = x.rolling_window(dim, window, 'window', center=center, fill_value=np.nan)
        with raise_if_dask_computes():
            actual = dx.rolling_window(dim, window, 'window', center=center, fill_value=np.nan)
        assert isinstance(actual.data, da.Array)
        assert actual.shape == expected.shape
        assert_equal(actual, expected)

    def test_multiindex(self):
        super().test_multiindex()

    @pytest.mark.parametrize('mode', ['mean', pytest.param('median', marks=pytest.mark.xfail(reason='median is not implemented by Dask')), pytest.param('reflect', marks=pytest.mark.xfail(reason='dask.array.pad bug')), 'edge', 'linear_ramp', 'maximum', 'minimum', 'symmetric', 'wrap'])
    @pytest.mark.parametrize('xr_arg, np_arg', _PAD_XR_NP_ARGS)
    @pytest.mark.filterwarnings('ignore:dask.array.pad.+? converts integers to floats.')
    def test_pad(self, mode, xr_arg, np_arg):
        super().test_pad(mode, xr_arg, np_arg)