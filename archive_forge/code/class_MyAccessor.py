from __future__ import annotations
import contextlib
import numpy as np
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_210, PANDAS_GE_300
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
class MyAccessor:

    def __init__(self, obj):
        self.obj = obj
        self.item = 'item'

    @property
    def prop(self):
        return self.item

    def method(self):
        return self.item