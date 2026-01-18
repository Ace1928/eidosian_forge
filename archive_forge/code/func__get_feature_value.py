import struct
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Union
import numpy as np
from ray.data.block import Block
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
def _get_feature_value(feature: 'tf.train.Feature', schema_feature_type: Optional['schema_pb2.FeatureType']=None) -> 'pyarrow.Array':
    import pyarrow as pa
    underlying_feature_type = {'bytes': feature.HasField('bytes_list'), 'float': feature.HasField('float_list'), 'int': feature.HasField('int64_list')}
    assert sum((bool(value) for value in underlying_feature_type.values())) <= 1
    if schema_feature_type is not None:
        try:
            from tensorflow_metadata.proto.v0 import schema_pb2
        except ModuleNotFoundError:
            raise ModuleNotFoundError('To use TensorFlow schemas, please install the tensorflow-metadata package.')
        specified_feature_type = {'bytes': schema_feature_type == schema_pb2.FeatureType.BYTES, 'float': schema_feature_type == schema_pb2.FeatureType.FLOAT, 'int': schema_feature_type == schema_pb2.FeatureType.INT}
        und_type = _get_single_true_type(underlying_feature_type)
        spec_type = _get_single_true_type(specified_feature_type)
        if und_type is not None and und_type != spec_type:
            raise ValueError(f'Schema field type mismatch during read: specified type is {spec_type}, but underlying type is {und_type}')
        underlying_feature_type = specified_feature_type
    if underlying_feature_type['bytes']:
        value = feature.bytes_list.value
        type_ = pa.binary()
    elif underlying_feature_type['float']:
        value = feature.float_list.value
        type_ = pa.float32()
    elif underlying_feature_type['int']:
        value = feature.int64_list.value
        type_ = pa.int64()
    else:
        value = []
        type_ = pa.null()
    value = list(value)
    if len(value) == 1 and schema_feature_type is None:
        value = value[0]
    else:
        if len(value) == 0:
            type_ = pa.null()
        type_ = pa.list_(type_)
    return pa.array([value], type=type_)