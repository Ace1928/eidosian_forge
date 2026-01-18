import argparse
import os
import sys
import warnings
from absl import app
import tensorflow as tf  # pylint: disable=unused-import
from tensorflow.lite.python import lite
from tensorflow.lite.python.convert import register_custom_opdefs
from tensorflow.lite.toco import toco_flags_pb2 as _toco_flags_pb2
from tensorflow.lite.toco.logging import gen_html
from tensorflow.python import tf2
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile
from tensorflow.python.util import keras_deps
def _get_parser(use_v2_converter):
    """Returns an ArgumentParser for tflite_convert.

  Args:
    use_v2_converter: Indicates which converter to return.
  Return: ArgumentParser.
  """
    parser = argparse.ArgumentParser(description='Command line tool to run TensorFlow Lite Converter.')
    parser.add_argument('--output_file', type=str, help='Full filepath of the output file.', required=True)
    if use_v2_converter:
        _get_tf2_flags(parser)
    else:
        _get_tf1_flags(parser)
    parser.add_argument('--experimental_new_converter', action=_ParseBooleanFlag, nargs='?', default=True, help='Experimental flag, subject to change. Enables MLIR-based conversion instead of TOCO conversion. (default True)')
    parser.add_argument('--experimental_new_quantizer', action=_ParseBooleanFlag, nargs='?', help='Experimental flag, subject to change. Enables MLIR-based quantizer instead of flatbuffer conversion. (default True)')
    return parser