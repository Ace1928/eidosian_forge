from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from dask.base import compute as dask_compute
from dask.dataframe import methods
from dask.dataframe._compat import PANDAS_GE_300
from dask.dataframe.io.io import from_delayed, from_pandas
from dask.dataframe.utils import pyarrow_strings_enabled
from dask.delayed import delayed, tokenize
from dask.utils import parse_bytes
def _to_sql_chunk(d, uri, engine_kwargs=None, **kwargs):
    import sqlalchemy as sa
    engine_kwargs = engine_kwargs or {}
    engine = sa.create_engine(uri, **engine_kwargs)
    q = d.to_sql(con=engine, **kwargs)
    engine.dispose()
    return q