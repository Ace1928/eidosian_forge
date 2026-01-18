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
class TestExtractDataset:

    def test_default(self):
        idata = load_arviz_data('centered_eight')
        post = extract(idata)
        assert isinstance(post, xr.Dataset)
        assert 'sample' in post.dims
        assert post.theta.size == 4 * 500 * 8

    def test_seed(self):
        idata = load_arviz_data('centered_eight')
        post = extract(idata, rng=7)
        post_pred = extract(idata, group='posterior_predictive', rng=7)
        assert all(post.sample == post_pred.sample)

    def test_no_combine(self):
        idata = load_arviz_data('centered_eight')
        post = extract(idata, combined=False)
        assert 'sample' not in post.dims
        assert post.sizes['chain'] == 4
        assert post.sizes['draw'] == 500

    def test_var_name_group(self):
        idata = load_arviz_data('centered_eight')
        prior = extract(idata, group='prior', var_names='the', filter_vars='like')
        assert {} == prior.attrs
        assert 'theta' in prior.name

    def test_keep_dataset(self):
        idata = load_arviz_data('centered_eight')
        prior = extract(idata, group='prior', var_names='the', filter_vars='like', keep_dataset=True)
        assert prior.attrs == idata.prior.attrs
        assert 'theta' in prior.data_vars
        assert 'mu' not in prior.data_vars

    def test_subset_samples(self):
        idata = load_arviz_data('centered_eight')
        post = extract(idata, num_samples=10)
        assert post.sizes['sample'] == 10
        assert post.attrs == idata.posterior.attrs