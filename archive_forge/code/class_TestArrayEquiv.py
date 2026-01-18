from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
class TestArrayEquiv:

    def test_0d(self):
        assert duck_array_ops.array_equiv(0, np.array(0, dtype=object))
        assert duck_array_ops.array_equiv(np.nan, np.array(np.nan, dtype=object))
        assert not duck_array_ops.array_equiv(0, np.array(1, dtype=object))