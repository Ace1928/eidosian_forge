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
@pytest.mark.skipif(not (importlib.util.find_spec('datatree') or running_on_ci()), reason='test requires xarray-datatree library')
class TestDataTree:

    def test_datatree(self):
        idata = load_arviz_data('centered_eight')
        dt = idata.to_datatree()
        idata_back = from_datatree(dt)
        for group, ds in idata.items():
            assert_identical(ds, idata_back[group])
        assert all((group in dt.children for group in idata.groups()))