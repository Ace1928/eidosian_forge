from __future__ import annotations
import math
import pickle
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import assert_equal, assert_identical, requires_dask
class TestSparseVariable:

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.data = sparse.random((4, 6), random_state=0, density=0.5)
        self.var = xr.Variable(('x', 'y'), self.data)

    def test_nbytes(self):
        assert self.var.nbytes == self.data.nbytes

    def test_unary_op(self):
        assert_sparse_equal(-self.var.data, -self.data)
        assert_sparse_equal(abs(self.var).data, abs(self.data))
        assert_sparse_equal(self.var.round().data, self.data.round())

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_univariate_ufunc(self):
        assert_sparse_equal(np.sin(self.data), np.sin(self.var).data)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_bivariate_ufunc(self):
        assert_sparse_equal(np.maximum(self.data, 0), np.maximum(self.var, 0).data)
        assert_sparse_equal(np.maximum(self.data, 0), np.maximum(0, self.var).data)

    def test_repr(self):
        expected = dedent('            <xarray.Variable (x: 4, y: 6)> Size: 288B\n            <COO: shape=(4, 6), dtype=float64, nnz=12, fill_value=0.0>')
        assert expected == repr(self.var)

    def test_pickle(self):
        v1 = self.var
        v2 = pickle.loads(pickle.dumps(v1))
        assert_sparse_equal(v1.data, v2.data)

    def test_missing_values(self):
        a = np.array([0, 1, np.nan, 3])
        s = sparse.COO.from_numpy(a)
        var_s = Variable('x', s)
        assert np.all(var_s.fillna(2).data.todense() == np.arange(4))
        assert np.all(var_s.count() == 3)