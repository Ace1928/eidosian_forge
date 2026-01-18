from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import copy, deepcopy
from io import StringIO
from textwrap import dedent
from typing import Any, Literal
import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.datetimes import DatetimeIndex
import xarray as xr
from xarray import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import dtypes, indexing, utils
from xarray.core.common import duck_array_ops, full_like
from xarray.core.coordinates import Coordinates, DatasetCoordinates
from xarray.core.indexes import Index, PandasIndex
from xarray.core.utils import is_scalar
from xarray.namedarray.pycompat import array_type, integer_types
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
class TestDropDuplicates:

    @pytest.mark.parametrize('keep', ['first', 'last', False])
    def test_drop_duplicates_1d(self, keep) -> None:
        ds = xr.Dataset({'a': ('time', [0, 5, 6, 7]), 'b': ('time', [9, 3, 8, 2])}, coords={'time': [0, 0, 1, 2]})
        if keep == 'first':
            a = [0, 6, 7]
            b = [9, 8, 2]
            time = [0, 1, 2]
        elif keep == 'last':
            a = [5, 6, 7]
            b = [3, 8, 2]
            time = [0, 1, 2]
        else:
            a = [6, 7]
            b = [8, 2]
            time = [1, 2]
        expected = xr.Dataset({'a': ('time', a), 'b': ('time', b)}, coords={'time': time})
        result = ds.drop_duplicates('time', keep=keep)
        assert_equal(expected, result)
        with pytest.raises(ValueError, match=re.escape("Dimensions ('space',) not found in data dimensions ('time',)")):
            ds.drop_duplicates('space', keep=keep)