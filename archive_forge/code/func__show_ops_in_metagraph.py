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
def _show_ops_in_metagraph(saved_model_dir, tag_set):
    """Prints the ops in the MetaGraph.

  Prints all the ops used in the MetaGraphDef indicated by the tag_set stored in
  SavedModel directory.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect.
    tag_set: Group of tag(s) of the MetaGraphDef in string format, separated by
      ','. For tag-set contains multiple tags, all tags must be passed in.
  """
    meta_graph_def = saved_model_utils.get_meta_graph_def(saved_model_dir, tag_set)
    _show_ops_in_metagraph_mgd(meta_graph_def)