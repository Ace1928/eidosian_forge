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
def _replace_colons(shapes, blob_name_tracker, ops, repl):
    """
    `:i` has a special meaning in Tensorflow. This function replaces all colons with $ to avoid any possible conflicts.

    Args:
        shapes: Dictionary mapping blob names to their shapes/dimensions.
        blob_name_tracker: Dictionary of all unique blob names (with respect to
            some context).
        ops: List of Caffe2 operators
        repl: String representing the text to replace ':' with. Usually this is
            '$'.

    Returns:
        None. Modifies blob_name_tracker in-place.

    """

    def f(name):
        return name.replace(':', repl)
    _rename_all(shapes, blob_name_tracker, ops, f)