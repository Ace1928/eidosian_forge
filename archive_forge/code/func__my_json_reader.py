from __future__ import annotations
import json
import os
import fsspec
import pandas as pd
import pytest
from packaging.version import Version
import dask
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200
from dask.dataframe.utils import assert_eq
from dask.utils import tmpdir, tmpfile
def _my_json_reader(*args, **kwargs):
    if fkeyword == 'json':
        return pd.DataFrame.from_dict(json.load(*args))
    return pd.read_json(*args)