from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def _check_box_coord(self, patches, expected_y=None, expected_h=None, expected_x=None, expected_w=None):
    result_y = np.array([p.get_y() for p in patches])
    result_height = np.array([p.get_height() for p in patches])
    result_x = np.array([p.get_x() for p in patches])
    result_width = np.array([p.get_width() for p in patches])
    if expected_y is not None:
        tm.assert_numpy_array_equal(result_y, expected_y, check_dtype=False)
    if expected_h is not None:
        tm.assert_numpy_array_equal(result_height, expected_h, check_dtype=False)
    if expected_x is not None:
        tm.assert_numpy_array_equal(result_x, expected_x, check_dtype=False)
    if expected_w is not None:
        tm.assert_numpy_array_equal(result_width, expected_w, check_dtype=False)