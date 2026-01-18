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
def _get_inputs_tensor_info_from_meta_graph_def(meta_graph_def, signature_def_key):
    """Gets TensorInfo for all inputs of the SignatureDef.

  Returns a dictionary that maps each input key to its TensorInfo for the given
  signature_def_key in the meta_graph_def

  Args:
    meta_graph_def: MetaGraphDef protocol buffer with the SignatureDef map to
        look up SignatureDef key.
    signature_def_key: A SignatureDef key string.

  Returns:
    A dictionary that maps input tensor keys to TensorInfos.

  Raises:
    ValueError if `signature_def_key` is not found in the MetaGraphDef.
  """
    if signature_def_key not in meta_graph_def.signature_def:
        raise ValueError(f'Could not find signature "{signature_def_key}". Please choose from: {', '.join(meta_graph_def.signature_def.keys())}')
    return meta_graph_def.signature_def[signature_def_key].inputs