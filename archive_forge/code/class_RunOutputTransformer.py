from typing import List, no_type_check
from triad import ParamDict, Schema, SerializableRLock, assert_or_throw
from triad.utils.convert import to_type
from fugue.collections.partition import PartitionCursor
from fugue.dataframe import DataFrame, DataFrames, LocalDataFrame
from fugue.dataframe.array_dataframe import ArrayDataFrame
from fugue.dataframe.utils import _df_eq
from fugue.exceptions import FugueWorkflowError
from fugue.execution.execution_engine import (
from fugue.rpc import EmptyRPCHandler, to_rpc_handler
from ..outputter import Outputter
from ..transformer.convert import _to_output_transformer
from ..transformer.transformer import CoTransformer, Transformer
class RunOutputTransformer(Outputter):

    @no_type_check
    def process(self, dfs: DataFrames) -> None:
        df = dfs[0]
        tf = _to_output_transformer(self.params.get_or_none('transformer', object))
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
            self.transform(df, tf)
        else:
            self.cotransform(df, tf)

    def transform(self, df: DataFrame, tf: Transformer) -> None:
        tf._key_schema = self.partition_spec.get_key_schema(df.schema)
        tf._output_schema = Schema(tf.get_output_schema(df))
        tr = _TransformerRunner(df, tf, self._ignore_errors)
        df = self.execution_engine.map_engine.map_dataframe(df=df, map_func=tr.run, output_schema=tf.output_schema, partition_spec=tf.partition_spec, on_init=tr.on_init, map_func_format_hint=tf.get_format_hint())
        self.execution_engine.persist(df, lazy=False)

    @no_type_check
    def cotransform(self, df: DataFrame, tf: CoTransformer) -> None:
        assert_or_throw(df.metadata.get('serialized', False), 'must use serialized dataframe')
        tf._key_schema = df.schema - _FUGUE_SERIALIZED_BLOB_SCHEMA
        empty_dfs = _generate_comap_empty_dfs(df.metadata['schemas'], df.metadata.get('serialized_has_name', False))
        tf._output_schema = Schema(tf.get_output_schema(empty_dfs))
        tr = _CoTransformerRunner(df, tf, self._ignore_errors)
        df = self.execution_engine.comap(df=df, map_func=tr.run, output_schema=tf.output_schema, partition_spec=tf.partition_spec, on_init=tr.on_init)
        self.execution_engine.persist(df, lazy=False)