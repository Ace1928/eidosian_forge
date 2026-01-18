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
class InaccessibleVariableDataStore(backends.InMemoryDataStore):
    """
    Store that does not allow any data access.
    """

    def __init__(self):
        super().__init__()
        self._indexvars = set()

    def store(self, variables, *args, **kwargs) -> None:
        super().store(variables, *args, **kwargs)
        for k, v in variables.items():
            if isinstance(v, IndexVariable):
                self._indexvars.add(k)

    def get_variables(self):

        def lazy_inaccessible(k, v):
            if k in self._indexvars:
                return v
            data = indexing.LazilyIndexedArray(InaccessibleArray(v.values))
            return Variable(v.dims, data, v.attrs)
        return {k: lazy_inaccessible(k, v) for k, v in self._variables.items()}