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
class TestCopyOnWriteArray:

    def test_setitem(self) -> None:
        original = np.arange(10)
        wrapped = indexing.CopyOnWriteArray(original)
        wrapped[B[:]] = 0
        assert_array_equal(original, np.arange(10))
        assert_array_equal(wrapped, np.zeros(10))

    def test_sub_array(self) -> None:
        original = np.arange(10)
        wrapped = indexing.CopyOnWriteArray(original)
        child = wrapped[B[:5]]
        assert isinstance(child, indexing.CopyOnWriteArray)
        child[B[:]] = 0
        assert_array_equal(original, np.arange(10))
        assert_array_equal(wrapped, np.arange(10))
        assert_array_equal(child, np.zeros(5))

    def test_index_scalar(self) -> None:
        x = indexing.CopyOnWriteArray(np.array(['foo', 'bar']))
        assert np.array(x[B[0]][B[()]]) == 'foo'