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
class SaveAndUse(Processor):

    def process(self, dfs: DataFrames) -> DataFrame:
        assert_or_throw(len(dfs) == 1, FugueWorkflowError('not single input'))
        kwargs = self.params.get('params', dict())
        path = self.params.get_or_throw('path', str)
        format_hint = self.params.get('fmt', '')
        mode = self.params.get('mode', 'overwrite')
        partition_spec = self.partition_spec
        force_single = self.params.get('single', False)
        self.execution_engine.save_df(df=dfs[0], path=path, format_hint=format_hint, mode=mode, partition_spec=partition_spec, force_single=force_single, **kwargs)
        return self.execution_engine.load_df(path=path, format_hint=format_hint)