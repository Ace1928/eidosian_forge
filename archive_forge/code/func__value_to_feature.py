import struct
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Union
import numpy as np
from ray.data.block import Block
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
def _value_to_feature(value: Union['pyarrow.Scalar', 'pyarrow.Array'], schema_feature_type: Optional['schema_pb2.FeatureType']=None) -> 'tf.train.Feature':
    import pyarrow as pa
    import tensorflow as tf
    if isinstance(value, pa.ListScalar):
        value_type = value.type.value_type
        value = value.as_py()
    else:
        value_type = value.type
        value = value.as_py()
        if value is None:
            value = []
        else:
            value = [value]
    underlying_value_type = {'bytes': pa.types.is_binary(value_type), 'string': pa.types.is_string(value_type), 'float': pa.types.is_floating(value_type), 'int': pa.types.is_integer(value_type)}
    assert sum((bool(value) for value in underlying_value_type.values())) <= 1
    if schema_feature_type is not None:
        try:
            from tensorflow_metadata.proto.v0 import schema_pb2
        except ModuleNotFoundError:
            raise ModuleNotFoundError('To use TensorFlow schemas, please install the tensorflow-metadata package.')
        specified_feature_type = {'bytes': schema_feature_type == schema_pb2.FeatureType.BYTES and (not underlying_value_type['string']), 'string': schema_feature_type == schema_pb2.FeatureType.BYTES and underlying_value_type['string'], 'float': schema_feature_type == schema_pb2.FeatureType.FLOAT, 'int': schema_feature_type == schema_pb2.FeatureType.INT}
        und_type = _get_single_true_type(underlying_value_type)
        spec_type = _get_single_true_type(specified_feature_type)
        if und_type is not None and und_type != spec_type:
            raise ValueError(f'Schema field type mismatch during write: specified type is {spec_type}, but underlying type is {und_type}')
        underlying_value_type = specified_feature_type
    if underlying_value_type['int']:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    if underlying_value_type['float']:
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    if underlying_value_type['bytes']:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    if underlying_value_type['string']:
        value = [v.encode() for v in value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    if pa.types.is_null(value_type):
        raise ValueError('Unable to infer type from partially missing column. Try setting read parallelism = 1, or use an input data source which explicitly specifies the schema.')
    raise ValueError(f'Value is of type {value_type}, which we cannot convert to a supported tf.train.Feature storage type (bytes, float, or int).')