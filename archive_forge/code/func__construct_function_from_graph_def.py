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
def _construct_function_from_graph_def(func, graph_def, frozen_func=None):
    """Rebuild function from graph_def."""
    if frozen_func is None:
        frozen_func = func
    for f in graph_def.library.function:
        while context.context().has_function(f.signature.name):
            context.context().remove_function(f.signature.name)
    captures = {c[1].name.split(':')[0]: c[0] for c in frozen_func.graph.captures}
    new_func = wrap_function.function_from_graph_def(graph_def, [tensor.name for tensor in frozen_func.inputs], [tensor.name for tensor in frozen_func.outputs], captures)
    new_func.graph.structured_outputs = nest.pack_sequence_as(func.graph.structured_outputs, new_func.graph.structured_outputs)
    new_func._function_type = func.function_type
    new_func.graph.structured_input_signature = func.structured_input_signature
    return new_func