import numpy as _np  # Avoids becoming a part of public Tensorflow API.
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
def _get_or_create_tuple_proto(self, op):
    try:
        attr = op.get_attr('_XlaSharding')
        proto = xla_data_pb2.OpSharding()
        proto.ParseFromString(attr)
        return proto
    except ValueError:
        return self._create_tuple_proto(len(op.outputs))