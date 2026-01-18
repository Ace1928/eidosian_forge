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
class DropColumns(Processor):

    def validate_on_compile(self):
        self.params.get_or_throw('columns', list)

    def process(self, dfs: DataFrames) -> DataFrame:
        assert_or_throw(len(dfs) == 1, FugueWorkflowError('not single input'))
        if_exists = self.params.get('if_exists', False)
        columns = self.params.get_or_throw('columns', list)
        if if_exists:
            columns = set(columns).intersection(dfs[0].schema.keys())
        return dfs[0].drop(list(columns))