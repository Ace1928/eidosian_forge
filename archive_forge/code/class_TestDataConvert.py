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
class TestDataConvert:

    @pytest.fixture(scope='class')
    def data(self, draws, chains):

        class Data:
            obj = {}
            for key, shape in {'mu': [], 'tau': [], 'eta': [8], 'theta': [8]}.items():
                obj[key] = np.random.randn(chains, draws, *shape)
        return Data

    def get_inference_data(self, data):
        return convert_to_inference_data(data.obj, group='posterior', coords={'school': np.arange(8)}, dims={'theta': ['school'], 'eta': ['school']})

    def check_var_names_coords_dims(self, dataset):
        assert set(dataset.data_vars) == {'mu', 'tau', 'eta', 'theta'}
        assert set(dataset.coords) == {'chain', 'draw', 'school'}

    def test_convert_to_inference_data(self, data):
        inference_data = self.get_inference_data(data)
        assert hasattr(inference_data, 'posterior')
        self.check_var_names_coords_dims(inference_data.posterior)

    def test_convert_to_dataset(self, draws, chains, data):
        dataset = convert_to_dataset(data.obj, group='posterior', coords={'school': np.arange(8)}, dims={'theta': ['school'], 'eta': ['school']})
        assert dataset.draw.shape == (draws,)
        assert dataset.chain.shape == (chains,)
        assert dataset.school.shape == (8,)
        assert dataset.theta.shape == (chains, draws, 8)