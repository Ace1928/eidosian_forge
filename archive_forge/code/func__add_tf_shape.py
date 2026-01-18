import copy
import logging
import os
import re
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from typing import Set, Dict, Tuple, List
def _add_tf_shape(attr_dict, ints):
    """
    Convert a list of ints to a TensorShapeProto representing the dimensions of a blob/object.

    Args:
        attr_dict: Dictionary to update (usually attributes of a Node)
        ints: List of integers representing dimensions of some object.

    Returns:
        None. Modifies attr_dict in-place.
    """
    shape_proto = TensorShapeProto()
    for i in ints:
        dim = TensorShapeProto.Dim()
        dim.size = i
        shape_proto.dim.extend([dim])
    attr_dict['_output_shapes'].list.shape.extend([shape_proto])