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
def _operator_to_node_simp(op, inter_blobs, seen):
    """
    Convert the operators to nodes.

    Args:
        op: Caffe2 operator to convert to node
        inter_blobs: Set of intermediate blobs
        seen: Names that have already been used and are not unique

    Returns:
        nodes: Nodes representing 'op' and the outputs of 'op'
    """
    assert op
    nodes = []
    outputs = [o for o in op.output if o not in inter_blobs]
    seen.update(outputs)
    len_outputs = len(outputs)
    if len_outputs == 1:
        n = NodeDef()
        n.name = outputs[0]
        n.input.extend(op.input)
        n.op = op.type
        n.device = _tf_device(op.device_option)
        for arg in op.arg:
            _set_tf_attr(n.attr, arg)
        nodes.append(n)
    elif len_outputs > 1:
        if op.name:
            name = op.name
        else:
            name_list = list(outputs)
            scope = os.path.commonprefix(name_list)
            name = os.path.join(scope, op.type)
        assert name
        op.name = _make_unique_name(seen, name)
        device = _tf_device(op.device_option)
        for output in outputs:
            n = NodeDef()
            n.name = output
            n.input.extend([op.name])
            n.op = 'Blob'
            n.device = device
            nodes.append(n)
        n = NodeDef()
        n.name = op.name
        n.input.extend(op.input)
        n.op = op.type
        n.device = device
        for arg in op.arg:
            _set_tf_attr(n.attr, arg)
        nodes.append(n)
    return nodes