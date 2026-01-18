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
class RunSetOperation(Processor):

    def process(self, dfs: DataFrames) -> DataFrame:
        if len(dfs) == 1:
            return dfs[0]
        how = self.params.get_or_throw('how', str)
        func: Any = {'union': self.execution_engine.union, 'subtract': self.execution_engine.subtract, 'intersect': self.execution_engine.intersect}[how]
        distinct = self.params.get('distinct', True)
        df = dfs[0]
        for i in range(1, len(dfs)):
            df = func(df, dfs[i], distinct=distinct)
        return df