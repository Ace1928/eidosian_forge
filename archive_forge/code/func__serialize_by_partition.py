import inspect
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from typing import (
from uuid import uuid4
from triad import ParamDict, Schema, SerializableRLock, assert_or_throw, to_uuid
from triad.collections.function_wrapper import AnnotatedParam
from triad.exceptions import InvalidOperationError
from triad.utils.convert import to_size
from fugue.bag import Bag, LocalBag
from fugue.collections.partition import (
from fugue.collections.sql import StructuredRawSQL, TempTableName
from fugue.collections.yielded import PhysicalYielded, Yielded
from fugue.column import (
from fugue.constants import _FUGUE_GLOBAL_CONF, FUGUE_SQL_DEFAULT_DIALECT
from fugue.dataframe import AnyDataFrame, DataFrame, DataFrames, fugue_annotated_param
from fugue.dataframe.array_dataframe import ArrayDataFrame
from fugue.dataframe.dataframe import LocalDataFrame
from fugue.dataframe.utils import deserialize_df, serialize_df
from fugue.exceptions import FugueWorkflowRuntimeError
def _serialize_by_partition(self, df: DataFrame, partition_spec: PartitionSpec, df_no: int, df_name: Optional[str], temp_path: Optional[str], to_file_threshold: Any) -> DataFrame:
    to_file_threshold = _get_file_threshold(to_file_threshold)
    on = list(filter(lambda k: k in df.schema, partition_spec.partition_by))
    presort = list(filter(lambda p: p[0] in df.schema, partition_spec.presort.items()))
    if len(on) == 0:
        _partition_spec = PartitionSpec(partition_spec, num=1, by=[], presort=presort)
        output_schema = _FUGUE_SERIALIZED_BLOB_SCHEMA
    else:
        _partition_spec = PartitionSpec(partition_spec, by=on, presort=presort)
        output_schema = partition_spec.get_key_schema(df.schema) + _FUGUE_SERIALIZED_BLOB_SCHEMA
    s = _PartitionSerializer(output_schema, df_no, df_name, temp_path, to_file_threshold)
    return self.map_engine.map_dataframe(df, s.run, output_schema, _partition_spec)