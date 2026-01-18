from __future__ import annotations
import itertools
from typing import Any
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable
from xarray.core import indexing, nputils
from xarray.core.indexes import PandasIndex, PandasMultiIndex
from xarray.core.types import T_Xarray
from xarray.tests import (
class TestMemoryCachedArray:

    def test_wrapper(self) -> None:
        original = indexing.LazilyIndexedArray(np.arange(10))
        wrapped = indexing.MemoryCachedArray(original)
        assert_array_equal(wrapped, np.arange(10))
        assert isinstance(wrapped.array, indexing.NumpyIndexingAdapter)

    def test_sub_array(self) -> None:
        original = indexing.LazilyIndexedArray(np.arange(10))
        wrapped = indexing.MemoryCachedArray(original)
        child = wrapped[B[:5]]
        assert isinstance(child, indexing.MemoryCachedArray)
        assert_array_equal(child, np.arange(5))
        assert isinstance(child.array, indexing.NumpyIndexingAdapter)
        assert isinstance(wrapped.array, indexing.LazilyIndexedArray)

    def test_setitem(self) -> None:
        original = np.arange(10)
        wrapped = indexing.MemoryCachedArray(original)
        wrapped[B[:]] = 0
        assert_array_equal(original, np.zeros(10))

    def test_index_scalar(self) -> None:
        x = indexing.MemoryCachedArray(np.array(['foo', 'bar']))
        assert np.array(x[B[0]][B[()]]) == 'foo'