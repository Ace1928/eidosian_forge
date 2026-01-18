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
class TestJSON:

    def test_json_converters(self, models):
        idata = models.model_1
        filepath = os.path.realpath('test.json')
        idata.to_json(filepath)
        idata_copy = from_json(filepath)
        for group in idata._groups_all:
            xr_data = getattr(idata, group)
            test_xr_data = getattr(idata_copy, group)
            assert xr_data.equals(test_xr_data)
        os.remove(filepath)
        assert not os.path.exists(filepath)