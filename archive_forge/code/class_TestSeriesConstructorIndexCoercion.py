from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
class TestSeriesConstructorIndexCoercion:

    def test_series_constructor_datetimelike_index_coercion(self):
        idx = date_range('2020-01-01', periods=5)
        ser = Series(np.random.default_rng(2).standard_normal(len(idx)), idx.astype(object))
        assert not isinstance(ser.index, DatetimeIndex)

    @pytest.mark.parametrize('container', [None, np.array, Series, Index])
    @pytest.mark.parametrize('data', [1.0, range(4)])
    def test_series_constructor_infer_multiindex(self, container, data):
        indexes = [['a', 'a', 'b', 'b'], ['x', 'y', 'x', 'y']]
        if container is not None:
            indexes = [container(ind) for ind in indexes]
        multi = Series(data, index=indexes)
        assert isinstance(multi.index, MultiIndex)