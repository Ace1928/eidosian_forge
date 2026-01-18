from __future__ import annotations
import codecs
from functools import wraps
import re
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._typing import (
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import ExtensionArray
from pandas.core.base import NoNewAttributesMixin
from pandas.core.construction import extract_array
def _get_series_list(self, others):
    """
        Auxiliary function for :meth:`str.cat`. Turn potentially mixed input
        into a list of Series (elements without an index must match the length
        of the calling Series/Index).

        Parameters
        ----------
        others : Series, DataFrame, np.ndarray, list-like or list-like of
            Objects that are either Series, Index or np.ndarray (1-dim).

        Returns
        -------
        list of Series
            Others transformed into list of Series.
        """
    from pandas import DataFrame, Series
    idx = self._orig if isinstance(self._orig, ABCIndex) else self._orig.index
    if isinstance(others, ABCSeries):
        return [others]
    elif isinstance(others, ABCIndex):
        return [Series(others, index=idx, dtype=others.dtype)]
    elif isinstance(others, ABCDataFrame):
        return [others[x] for x in others]
    elif isinstance(others, np.ndarray) and others.ndim == 2:
        others = DataFrame(others, index=idx)
        return [others[x] for x in others]
    elif is_list_like(others, allow_sets=False):
        try:
            others = list(others)
        except TypeError:
            pass
        else:
            if all((isinstance(x, (ABCSeries, ABCIndex, ExtensionArray)) or (isinstance(x, np.ndarray) and x.ndim == 1) for x in others)):
                los: list[Series] = []
                while others:
                    los = los + self._get_series_list(others.pop(0))
                return los
            elif all((not is_list_like(x) for x in others)):
                return [Series(others, index=idx)]
    raise TypeError('others must be Series, Index, DataFrame, np.ndarray or list-like (either containing only strings or containing only objects of type Series/Index/np.ndarray[1-dim])')