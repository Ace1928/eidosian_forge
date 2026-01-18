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
def _set_tf_attr(attr_dict, arg):
    """
    Add attributes to a node. Key is the arg.name, and values can be shape, floats, strings, ints or an empty list.

    Args:
        attr_dict: Dictionary to update (usually attributes of a Node)
        arg: Object with name and data fields.

    Returns:
        None. Modifies attr_dict in-place.
    """
    k = arg.name
    if k == 'shape' and arg.ints:
        _add_tf_shape(attr_dict, arg.ints)
        return
    if arg.HasField('f'):
        attr_dict[k].f = arg.f
        return
    if arg.HasField('i'):
        attr_dict[k].i = arg.i
        return
    if arg.HasField('s'):
        attr_dict[k].s = arg.s if isinstance(arg.s, bytes) else str(arg.s).encode('utf-8')
        return
    if arg.floats:
        attr_dict[k].list.f.extend(arg.floats)
        return
    if arg.ints:
        attr_dict[k].list.i.extend(arg.ints)
        return
    if arg.strings:
        attr_dict[k].list.s.extend((s if isinstance(s, bytes) else str(s).encode('utf-8') for s in arg.strings))
        return
    attr_dict[k].list.s.extend([])