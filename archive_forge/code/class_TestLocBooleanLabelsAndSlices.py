from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
class TestLocBooleanLabelsAndSlices:

    @pytest.mark.parametrize('bool_value', [True, False])
    def test_loc_bool_incompatible_index_raises(self, index, frame_or_series, bool_value):
        message = f'{bool_value}: boolean label can not be used without a boolean index'
        if index.inferred_type != 'boolean':
            obj = frame_or_series(index=index, dtype='object')
            with pytest.raises(KeyError, match=message):
                obj.loc[bool_value]

    @pytest.mark.parametrize('bool_value', [True, False])
    def test_loc_bool_should_not_raise(self, frame_or_series, bool_value):
        obj = frame_or_series(index=Index([True, False], dtype='boolean'), dtype='object')
        obj.loc[bool_value]

    def test_loc_bool_slice_raises(self, index, frame_or_series):
        message = 'slice\\(True, False, None\\): boolean values can not be used in a slice'
        obj = frame_or_series(index=index, dtype='object')
        with pytest.raises(TypeError, match=message):
            obj.loc[True:False]