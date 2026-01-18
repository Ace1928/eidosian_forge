from typing import Any
import dask.dataframe as dd
from dask.distributed import Client
from fugue import DataFrame
from fugue.dev import (
from fugue.plugins import (
from fugue_dask._utils import DASK_UTILS
from fugue_dask.dataframe import DaskDataFrame
from fugue_dask.execution_engine import DaskExecutionEngine
@parse_execution_engine.candidate(lambda engine, conf, **kwargs: isinstance(engine, str) and engine == 'dask', priority=4)
def _parse_dask_str(engine: str, conf: Any, **kwargs: Any) -> DaskExecutionEngine:
    return DaskExecutionEngine(conf=conf)