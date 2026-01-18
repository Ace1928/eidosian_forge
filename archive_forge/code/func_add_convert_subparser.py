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
def add_convert_subparser(subparsers):
    """Add parser for `convert`."""
    convert_msg = 'Usage example:\nTo convert the SavedModel to one that have TensorRT ops:\n$saved_model_cli convert \\\n   --dir /tmp/saved_model \\\n   --tag_set serve \\\n   --output_dir /tmp/saved_model_trt \\\n   tensorrt \n'
    parser_convert = subparsers.add_parser('convert', description=convert_msg, formatter_class=argparse.RawTextHelpFormatter)
    convert_subparsers = parser_convert.add_subparsers(title='conversion methods', description='valid conversion methods', help='the conversion to run with the SavedModel')
    parser_convert_with_tensorrt = convert_subparsers.add_parser('tensorrt', description='Convert the SavedModel with Tensorflow-TensorRT integration', formatter_class=argparse.RawTextHelpFormatter)
    parser_convert_with_tensorrt.set_defaults(func=convert_with_tensorrt)