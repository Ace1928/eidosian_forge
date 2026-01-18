import struct
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Union
import numpy as np
from ray.data.block import Block
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
def _convert_arrow_table_to_examples(arrow_table: 'pyarrow.Table', tf_schema: Optional['schema_pb2.Schema']=None) -> Iterable['tf.train.Example']:
    import tensorflow as tf
    schema_dict = {}
    if tf_schema is not None:
        for schema_feature in tf_schema.feature:
            schema_dict[schema_feature.name] = schema_feature.type
    for i in range(arrow_table.num_rows):
        features: Dict[str, 'tf.train.Feature'] = {}
        for name in arrow_table.column_names:
            if tf_schema is not None and name not in schema_dict:
                raise ValueError(f'Found extra unexpected feature {name} not in specified schema: {tf_schema}')
            schema_feature_type = schema_dict.get(name)
            features[name] = _value_to_feature(arrow_table[name][i], schema_feature_type)
        proto = tf.train.Example(features=tf.train.Features(feature=features))
        yield proto