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
class Fillna(Processor):

    def process(self, dfs: DataFrames) -> DataFrame:
        assert_or_throw(len(dfs) == 1, FugueWorkflowError('not single input'))
        value = self.params.get_or_none('value', object)
        assert_or_throw(not isinstance(value, list) and value is not None, FugueWorkflowError('fillna value cannot be None or list'))
        if isinstance(value, dict):
            assert_or_throw(None not in value.values() and any(value.values()), FugueWorkflowError("fillna dict can't contain None and must have len > 1"))
        subset = self.params.get_or_none('subset', list)
        return self.execution_engine.fillna(dfs[0], value=value, subset=subset)