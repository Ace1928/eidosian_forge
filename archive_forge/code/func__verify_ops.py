import collections
import os
import re
import sys
import traceback
from typing import Any, Callable, Dict, List, Tuple
from absl import logging
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import util as checkpoint_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.eager.polymorphic_function import concrete_function as cf
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.eager.polymorphic_function import saved_model_exported_concrete
from tensorflow.python.eager.polymorphic_function import saved_model_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function as framework_fn
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import versions
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import builder_impl
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import function_serialization
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import pywrap_saved_model
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import signature_serialization
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import tracing_utils
from tensorflow.python.saved_model import utils_impl
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base
from tensorflow.python.trackable import resource
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import trace_saveable_util
from tensorflow.python.types import core as types_core
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_stack
from tensorflow.python.util.tf_export import tf_export
def _verify_ops(graph_def: graph_pb2.GraphDef, namespace_whitelist):
    """Verifies that all namespaced ops in the graph are whitelisted.

  Args:
   graph_def: the GraphDef to validate.
   namespace_whitelist: a list of namespaces to allow. If `None`, all will be
     allowed. If an op does not have a namespace, it will be allowed.

  Raises:
   ValueError: If the graph contains ops that violate the whitelist.
  """
    if namespace_whitelist is None:
        return
    invalid_ops = []
    invalid_namespaces = set()
    all_operations = []
    all_operations.extend(meta_graph.ops_used_by_graph_def(graph_def))
    for op in all_operations:
        if '>' in op:
            namespace = op.split('>')[0]
            if namespace not in namespace_whitelist:
                invalid_ops.append(op)
                invalid_namespaces.add(namespace)
    if invalid_ops:
        raise ValueError(f"Attempted to save ops from non-whitelisted namespaces to SavedModel: {invalid_ops}.\nPlease verify that these ops should be saved, since they must be available when loading the SavedModel. If loading from Python, you must import the library defining these ops. From C++, link the custom ops to the serving binary. Once you've confirmed this, add the following namespaces to the `namespace_whitelist` argument in tf.saved_model.SaveOptions: {invalid_namespaces}.")