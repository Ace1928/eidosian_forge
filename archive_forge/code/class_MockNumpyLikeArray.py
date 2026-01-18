import collections
from collections import namedtuple
from collections.abc import Iterator
from datetime import (
from decimal import Decimal
from fractions import Fraction
from io import StringIO
import itertools
from numbers import Number
import re
import sys
from typing import (
import numpy as np
import pytest
import pytz
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes import inference
from pandas.core.dtypes.cast import find_result_type
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
class MockNumpyLikeArray:
    """
    A class which is numpy-like (e.g. Pint's Quantity) but not actually numpy

    The key is that it is not actually a numpy array so
    ``util.is_array(mock_numpy_like_array_instance)`` returns ``False``. Other
    important properties are that the class defines a :meth:`__iter__` method
    (so that ``isinstance(abc.Iterable)`` returns ``True``) and has a
    :meth:`ndim` property, as pandas special-cases 0-dimensional arrays in some
    cases.

    We expect pandas to behave with respect to such duck arrays exactly as
    with real numpy arrays. In particular, a 0-dimensional duck array is *NOT*
    a scalar (`is_scalar(np.array(1)) == False`), but it is not list-like either.
    """

    def __init__(self, values) -> None:
        self._values = values

    def __iter__(self) -> Iterator:
        iter_values = iter(self._values)

        def it_outer():
            yield from iter_values
        return it_outer()

    def __len__(self) -> int:
        return len(self._values)

    def __array__(self, t=None):
        return np.asarray(self._values, dtype=t)

    @property
    def ndim(self):
        return self._values.ndim

    @property
    def dtype(self):
        return self._values.dtype

    @property
    def size(self):
        return self._values.size

    @property
    def shape(self):
        return self._values.shape