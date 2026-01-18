from typing import Any, Callable
import dask.dataframe as dd
import ibis
from ibis.backends.dask import Backend
from triad.utils.assertion import assert_or_throw
from fugue import DataFrame, DataFrames, ExecutionEngine
from fugue_dask.dataframe import DaskDataFrame
from fugue_dask.execution_engine import DaskExecutionEngine
from fugue_ibis import IbisTable
from fugue_ibis._utils import to_ibis_schema, to_schema
from fugue_ibis.execution.ibis_engine import IbisEngine, parse_ibis_engine
@parse_ibis_engine.candidate(lambda obj, *args, **kwargs: isinstance(obj, DaskExecutionEngine))
def _to_dask_ibis_engine(obj: Any, engine: ExecutionEngine) -> IbisEngine:
    return DaskIbisEngine(engine)