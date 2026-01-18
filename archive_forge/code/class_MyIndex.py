import datetime
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
class MyIndex(pd.Index):
    _calls: int

    @classmethod
    def _simple_new(cls, values, name=None, dtype=None):
        result = object.__new__(cls)
        result._data = values
        result._name = name
        result._calls = 0
        result._reset_identity()
        return result

    def __add__(self, other):
        self._calls += 1
        return self._simple_new(self._data)

    def __radd__(self, other):
        return self.__add__(other)