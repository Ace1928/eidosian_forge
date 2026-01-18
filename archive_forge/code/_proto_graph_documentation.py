from typing import Optional
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
Create an object matching a NodeDef.

    Follows https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/node_def.proto .
    