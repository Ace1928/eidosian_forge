from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
class Test_hashable:

    def test_hashable(self):
        for v in [False, 1, (2,), (3, 4), 'four']:
            assert utils.hashable(v)
        for v in [[5, 6], ['seven', '8'], {9: 'ten'}]:
            assert not utils.hashable(v)