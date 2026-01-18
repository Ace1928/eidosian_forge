import argparse
import platform
import ast
import os
import re
from absl import app  # pylint: disable=unused-import
from absl import flags
from absl.flags import argparse_flags
import numpy as np
from tensorflow.core.example import example_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.framework import meta_graph as meta_graph_lib
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.tools import saved_model_aot_compile
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.tpu import tpu
from tensorflow.python.util.compat import collections_abc
def _print_tensor_info(tensor_info, indent=0):
    """Prints details of the given tensor_info.

  Args:
    tensor_info: TensorInfo object to be printed.
    indent: How far (in increments of 2 spaces) to indent each line output
  """
    indent_str = '  ' * indent

    def in_print(s):
        print(indent_str + s)
    in_print('    dtype: ' + {value: key for key, value in types_pb2.DataType.items()}[tensor_info.dtype])
    if tensor_info.tensor_shape.unknown_rank:
        shape = 'unknown_rank'
    else:
        dims = [str(dim.size) for dim in tensor_info.tensor_shape.dim]
        shape = ', '.join(dims)
        shape = '(' + shape + ')'
    in_print('    shape: ' + shape)
    in_print('    name: ' + tensor_info.name)