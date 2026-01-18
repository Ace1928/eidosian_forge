import functools
import os
from google.protobuf.any_pb2 import Any
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def _maybe_add_main_op(self, main_op):
    """Adds main op to the SavedModel.

    Args:
      main_op: Main op to run as part of graph initialization. If None, no main
        op will be added to the graph.

    Raises:
      TypeError: If the main op is provided but is not of type `Operation`.
      ValueError: if the Graph already contains an init op.
    """
    if main_op is None:
        return
    if not isinstance(main_op, ops.Operation):
        raise TypeError(f'Expected {main_op} to be an Operation but got type {type(main_op)} instead.')
    for init_op_key in (constants.MAIN_OP_KEY, constants.LEGACY_INIT_OP_KEY):
        if ops.get_collection(init_op_key):
            raise ValueError(f'Graph already contains one or more main ops under the collection {init_op_key}.')
    ops.add_to_collection(constants.MAIN_OP_KEY, main_op)