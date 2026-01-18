import copy
from packaging import version as packaging_version  # pylint: disable=g-bad-import-order
import os.path
import re
import sys
from google.protobuf.any_pb2 import Any
from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import context
from tensorflow.python.framework import byte_swap_tensor as bst
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import importer
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import versions
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def _get_kind_name(item):
    """Returns the kind name in CollectionDef.

  Args:
    item: A data item.

  Returns:
    The string representation of the kind in CollectionDef.
  """
    if isinstance(item, (str, bytes)):
        kind = 'bytes_list'
    elif isinstance(item, int):
        kind = 'int64_list'
    elif isinstance(item, float):
        kind = 'float_list'
    elif isinstance(item, Any):
        kind = 'any_list'
    else:
        kind = 'node_list'
    return kind