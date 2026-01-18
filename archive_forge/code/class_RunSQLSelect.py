from typing import Any, List, Type, no_type_check
from triad.collections import ParamDict
from triad.collections.schema import Schema
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import to_type
from fugue.collections.partition import PartitionCursor
from fugue.column import ColumnExpr
from fugue.column import SelectColumns as ColumnsSelect
from fugue.dataframe import ArrayDataFrame, DataFrame, DataFrames, LocalDataFrame
from fugue.exceptions import FugueWorkflowError
from fugue.execution import make_sql_engine
from fugue.execution.execution_engine import (
from fugue.extensions.processor import Processor
from fugue.extensions.transformer import CoTransformer, Transformer, _to_transformer
from fugue.rpc import EmptyRPCHandler, to_rpc_handler
class RunSQLSelect(Processor):

    def process(self, dfs: DataFrames) -> DataFrame:
        statement = self.params.get_or_throw('statement', object)
        engine = self.params.get_or_none('sql_engine', object)
        engine_params = self.params.get('sql_engine_params', ParamDict())
        sql_engine = make_sql_engine(engine, self.execution_engine, **engine_params)
        return sql_engine.select(dfs, statement)