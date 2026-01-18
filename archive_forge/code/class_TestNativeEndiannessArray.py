from __future__ import annotations
import contextlib
import warnings
import numpy as np
import pandas as pd
import pytest
from xarray import (
from xarray.backends.common import WritableCFDataStore
from xarray.backends.memory import InMemoryDataStore
from xarray.conventions import decode_cf
from xarray.testing import assert_identical
from xarray.tests import (
from xarray.tests.test_backends import CFEncodedBase
class TestNativeEndiannessArray:

    def test(self) -> None:
        x = np.arange(5, dtype='>i8')
        expected = np.arange(5, dtype='int64')
        a = coding.variables.NativeEndiannessArray(x)
        assert a.dtype == expected.dtype
        assert a.dtype == expected[:].dtype
        assert_array_equal(a, expected)