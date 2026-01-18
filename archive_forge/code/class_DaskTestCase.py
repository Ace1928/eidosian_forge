from __future__ import annotations
import operator
import pickle
import sys
from contextlib import suppress
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core import duck_array_ops
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.testing import assert_chunks_equal
from xarray.tests import (
from xarray.tests.test_backends import create_tmp_file
class DaskTestCase:

    def assertLazyAnd(self, expected, actual, test):
        with dask.config.set(scheduler='synchronous'):
            test(actual, expected)
        if isinstance(actual, Dataset):
            for k, v in actual.variables.items():
                if k in actual.xindexes:
                    assert isinstance(v.data, np.ndarray)
                else:
                    assert isinstance(v.data, da.Array)
        elif isinstance(actual, DataArray):
            assert isinstance(actual.data, da.Array)
            for k, v in actual.coords.items():
                if k in actual.xindexes:
                    assert isinstance(v.data, np.ndarray)
                else:
                    assert isinstance(v.data, da.Array)
        elif isinstance(actual, Variable):
            assert isinstance(actual.data, da.Array)
        else:
            assert False