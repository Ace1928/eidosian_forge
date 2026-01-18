from __future__ import annotations
import re
import string
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, cast
import numpy as np
import pandas as pd
from dask.dataframe._compat import PANDAS_GE_220, PANDAS_GE_300
from dask.dataframe._pyarrow import is_object_string_dtype
from dask.dataframe.core import tokenize
from dask.dataframe.io.utils import DataFrameIOFunction
from dask.utils import random_state_data
class MakeDataframePart(DataFrameIOFunction):
    """
    Wrapper Class for ``make_dataframe_part``
    Makes a timeseries partition.
    """

    def __init__(self, index_dtype, dtypes, kwargs, columns=None):
        self.index_dtype = index_dtype
        self._columns = columns or list(dtypes.keys())
        self.dtypes = dtypes
        self.kwargs = kwargs

    @property
    def columns(self):
        return self._columns

    def project_columns(self, columns):
        """Return a new MakeTimeseriesPart object with
        a sub-column projection.
        """
        if columns == self.columns:
            return self
        return MakeDataframePart(self.index_dtype, self.dtypes, self.kwargs, columns=columns)

    def __call__(self, part):
        divisions, state_data = part
        return make_dataframe_part(self.index_dtype, divisions[0], divisions[1], self.dtypes, self.columns, state_data, self.kwargs)