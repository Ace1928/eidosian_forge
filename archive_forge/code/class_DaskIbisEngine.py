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
class DaskIbisEngine(IbisEngine):

    def __init__(self, execution_engine: ExecutionEngine) -> None:
        assert_or_throw(isinstance(execution_engine, DaskExecutionEngine), lambda: ValueError(f'DaskIbisEngine must use DaskExecutionEngine ({execution_engine})'))
        super().__init__(execution_engine)

    def select(self, dfs: DataFrames, ibis_func: Callable[[ibis.BaseBackend], IbisTable]) -> DataFrame:
        pdfs = {k: self.execution_engine.to_df(v).native for k, v in dfs.items()}
        be = _BackendWrapper().connect(pdfs)
        be.set_schemas(dfs)
        expr = ibis_func(be)
        schema = to_schema(expr.schema())
        result = expr.compile()
        assert_or_throw(isinstance(result, dd.DataFrame), lambda: ValueError(f'result must be a Dask DataFrame ({type(result)})'))
        return DaskDataFrame(result, schema=schema)