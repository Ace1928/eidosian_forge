import collections
from functools import partial  # pylint: disable=g-importing-member
import os
import platform
import sys
import tempfile
import numpy as np
import six as _six
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import resource
from tensorflow.python.training import saver
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
def _annotate_variable_ops(func, graph_def):
    """Annotates variable operations with custom `_shape` attribute.

  This is required for the converters and shape inference. The graph
  definition is modified in-place.

  Args:
    func: Function represented by the graph definition.
    graph_def: Graph definition to be annotated in-place.

  Raises:
    RuntimeError: if some shapes cannot be annotated.
  """
    ph_shape_map = {}
    for ph, var in zip(func.graph.internal_captures, func.variables):
        ph_shape_map[ph.name] = var.shape
    name_to_node = {node.name: node for node in graph_def.node}
    for node in graph_def.node:
        if node.op == 'ReadVariableOp' or node.op == 'ResourceGather':
            node_ = node
            while name_to_node[node_.input[0]].op == 'Identity':
                node_ = name_to_node[node_.input[0]]
            ph_name = node_.input[0] + ':0'
            if ph_name in ph_shape_map:
                shape = ph_shape_map[ph_name]
                node.attr['_shape'].shape.CopyFrom(shape.as_proto())
            else:
                raise RuntimeError('Not found in the function captures: {}'.format(ph_name))