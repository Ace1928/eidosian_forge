from __future__ import annotations
import contextlib
import decimal
import warnings
import weakref
import xml.etree.ElementTree
from datetime import datetime, timedelta
from itertools import product
from operator import add
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from pandas.errors import PerformanceWarning
from pandas.io.formats import format as pandas_format
import dask
import dask.array as da
import dask.dataframe as dd
import dask.dataframe.groupby
from dask import delayed
from dask.base import compute_as_if_collection
from dask.blockwise import fuse_roots
from dask.dataframe import _compat, methods
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.core import (
from dask.dataframe.utils import (
from dask.datasets import timeseries
from dask.utils import M, is_dataframe_like, is_series_like, put_lines
from dask.utils_test import _check_warning, hlg_layer
def _assert_info(df, ddf, memory_usage=True):
    from io import StringIO
    assert isinstance(df, pd.DataFrame)
    assert isinstance(ddf, dd.DataFrame)
    buf_pd, buf_da = (StringIO(), StringIO())
    df.info(buf=buf_pd, memory_usage=memory_usage)
    ddf.info(buf=buf_da, verbose=True, memory_usage=memory_usage)
    stdout_pd = buf_pd.getvalue()
    stdout_da = buf_da.getvalue()
    stdout_da = stdout_da.replace(str(type(ddf)), str(type(df)))
    assert stdout_pd == stdout_da