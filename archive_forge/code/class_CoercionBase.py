from __future__ import annotations
from datetime import (
import itertools
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas as pd
import pandas._testing as tm
class CoercionBase:
    klasses = ['index', 'series']
    dtypes = ['object', 'int64', 'float64', 'complex128', 'bool', 'datetime64', 'datetime64tz', 'timedelta64', 'period']

    @property
    def method(self):
        raise NotImplementedError(self)