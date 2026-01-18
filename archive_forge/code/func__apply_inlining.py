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
def _apply_inlining(func):
    """Apply an inlining optimization to the function's graph definition."""
    graph_def = func.graph.as_graph_def()
    for function in graph_def.library.function:
        if 'api_implements' in function.attr:
            del function.attr['api_implements']
    meta_graph = saver.export_meta_graph(graph_def=graph_def, graph=func.graph)
    for name in ['variables', 'model_variables', 'trainable_variables', 'local_variables']:
        raw_list = []
        for raw in meta_graph.collection_def['variables'].bytes_list.value:
            variable = variable_pb2.VariableDef()
            variable.ParseFromString(raw)
            variable.ClearField('initializer_name')
            raw_list.append(variable.SerializeToString())
        meta_graph.collection_def[name].bytes_list.value[:] = raw_list
    fetch_collection = meta_graph_pb2.CollectionDef()
    for array in func.inputs + func.outputs:
        fetch_collection.node_list.value.append(array.name)
    meta_graph.collection_def['train_op'].CopyFrom(fetch_collection)
    config = config_pb2.ConfigProto()
    rewrite_options = config.graph_options.rewrite_options
    rewrite_options.min_graph_nodes = -1
    rewrite_options.optimizers.append('function')
    new_graph_def = tf_optimizer.OptimizeGraph(config, meta_graph)
    return new_graph_def