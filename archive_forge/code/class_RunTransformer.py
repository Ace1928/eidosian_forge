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
class RunTransformer(Processor):

    @no_type_check
    def process(self, dfs: DataFrames) -> DataFrame:
        df = dfs[0]
        tf = _to_transformer(self.params.get_or_none('transformer', object), self.params.get_or_none('schema', object))
        tf._workflow_conf = self.execution_engine.conf
        tf._params = self.params.get('params', ParamDict())
        tf._partition_spec = self.partition_spec
        rpc_handler = to_rpc_handler(self.params.get_or_throw('rpc_handler', object))
        if not isinstance(rpc_handler, EmptyRPCHandler):
            tf._rpc_client = self.rpc_server.make_client(rpc_handler)
        ie = self.params.get('ignore_errors', [])
        self._ignore_errors = [to_type(x, Exception) for x in ie]
        tf.validate_on_runtime(df)
        if isinstance(tf, Transformer):
            return self.transform(df, tf)
        else:
            return self.cotransform(df, tf)

    def transform(self, df: DataFrame, tf: Transformer) -> DataFrame:
        tf._key_schema = self.partition_spec.get_key_schema(df.schema)
        tf._output_schema = Schema(tf.get_output_schema(df))
        tr = _TransformerRunner(df, tf, self._ignore_errors)
        return self.execution_engine.map_engine.map_dataframe(df=df, map_func=tr.run, output_schema=tf.output_schema, partition_spec=tf.partition_spec, on_init=tr.on_init, map_func_format_hint=tf.get_format_hint())

    @no_type_check
    def cotransform(self, df: DataFrame, tf: CoTransformer) -> DataFrame:
        assert_or_throw(df.metadata.get('serialized', False), 'must use serialized dataframe')
        tf._key_schema = df.schema - _FUGUE_SERIALIZED_BLOB_SCHEMA
        empty_dfs = _generate_comap_empty_dfs(df.metadata['schemas'], df.metadata.get('serialized_has_name', False))
        tf._output_schema = Schema(tf.get_output_schema(empty_dfs))
        tr = _CoTransformerRunner(df, tf, self._ignore_errors)
        return self.execution_engine.comap(df=df, map_func=tr.run, output_schema=tf.output_schema, partition_spec=tf.partition_spec, on_init=tr.on_init)