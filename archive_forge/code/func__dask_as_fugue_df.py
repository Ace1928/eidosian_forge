from typing import Any
import dask.dataframe as dd
from dask.distributed import Client
from fugue import DataFrame
from fugue.dev import (
from fugue.plugins import (
from fugue_dask._utils import DASK_UTILS
from fugue_dask.dataframe import DaskDataFrame
from fugue_dask.execution_engine import DaskExecutionEngine
@as_fugue_dataset.candidate(lambda df, **kwargs: isinstance(df, dd.DataFrame))
def _dask_as_fugue_df(df: dd.DataFrame, **kwargs: Any) -> DaskDataFrame:
    return DaskDataFrame(df, **kwargs)