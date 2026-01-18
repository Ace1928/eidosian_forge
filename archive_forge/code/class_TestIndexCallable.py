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
class TestIndexCallable:

    def test_getitem(self):

        def getter(key):
            return key * 2
        indexer = indexing.IndexCallable(getter)
        assert indexer[3] == 6
        assert indexer[0] == 0
        assert indexer[-1] == -2

    def test_setitem(self):

        def getter(key):
            return key * 2

        def setter(key, value):
            raise NotImplementedError('Setter not implemented')
        indexer = indexing.IndexCallable(getter, setter)
        with pytest.raises(NotImplementedError):
            indexer[3] = 6