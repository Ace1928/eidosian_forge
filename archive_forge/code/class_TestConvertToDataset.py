import importlib
import os
from collections import namedtuple
from copy import deepcopy
from html import escape
from typing import Dict
from tempfile import TemporaryDirectory
from urllib.parse import urlunsplit
import numpy as np
import pytest
import xarray as xr
from xarray.core.options import OPTIONS
from xarray.testing import assert_identical
from ... import (
from ...data.base import dict_to_dataset, generate_dims_coords, infer_stan_dtypes, make_attrs
from ...data.datasets import LOCAL_DATASETS, REMOTE_DATASETS, RemoteFileMetadata
from ..helpers import (  # pylint: disable=unused-import
class TestConvertToDataset:

    @pytest.fixture(scope='class')
    def data(self):

        class Data:
            datadict = {'a': np.random.randn(100), 'b': np.random.randn(1, 100, 10), 'c': np.random.randn(1, 100, 3, 4)}
            coords = {'c1': np.arange(3), 'c2': np.arange(4), 'b1': np.arange(10)}
            dims = {'b': ['b1'], 'c': ['c1', 'c2']}
        return Data

    def test_use_all(self, data):
        dataset = convert_to_dataset(data.datadict, coords=data.coords, dims=data.dims)
        assert set(dataset.data_vars) == {'a', 'b', 'c'}
        assert set(dataset.coords) == {'chain', 'draw', 'c1', 'c2', 'b1'}
        assert set(dataset.a.coords) == {'chain', 'draw'}
        assert set(dataset.b.coords) == {'chain', 'draw', 'b1'}
        assert set(dataset.c.coords) == {'chain', 'draw', 'c1', 'c2'}

    def test_missing_coords(self, data):
        dataset = convert_to_dataset(data.datadict, coords=None, dims=data.dims)
        assert set(dataset.data_vars) == {'a', 'b', 'c'}
        assert set(dataset.coords) == {'chain', 'draw', 'c1', 'c2', 'b1'}
        assert set(dataset.a.coords) == {'chain', 'draw'}
        assert set(dataset.b.coords) == {'chain', 'draw', 'b1'}
        assert set(dataset.c.coords) == {'chain', 'draw', 'c1', 'c2'}

    def test_missing_dims(self, data):
        coords = {'c_dim_0': np.arange(3), 'c_dim_1': np.arange(4), 'b_dim_0': np.arange(10)}
        dataset = convert_to_dataset(data.datadict, coords=coords, dims=None)
        assert set(dataset.data_vars) == {'a', 'b', 'c'}
        assert set(dataset.coords) == {'chain', 'draw', 'c_dim_0', 'c_dim_1', 'b_dim_0'}
        assert set(dataset.a.coords) == {'chain', 'draw'}
        assert set(dataset.b.coords) == {'chain', 'draw', 'b_dim_0'}
        assert set(dataset.c.coords) == {'chain', 'draw', 'c_dim_0', 'c_dim_1'}

    def test_skip_dim_0(self, data):
        dims = {'c': [None, 'c2']}
        coords = {'c_dim_0': np.arange(3), 'c2': np.arange(4), 'b_dim_0': np.arange(10)}
        dataset = convert_to_dataset(data.datadict, coords=coords, dims=dims)
        assert set(dataset.data_vars) == {'a', 'b', 'c'}
        assert set(dataset.coords) == {'chain', 'draw', 'c_dim_0', 'c2', 'b_dim_0'}
        assert set(dataset.a.coords) == {'chain', 'draw'}
        assert set(dataset.b.coords) == {'chain', 'draw', 'b_dim_0'}
        assert set(dataset.c.coords) == {'chain', 'draw', 'c_dim_0', 'c2'}