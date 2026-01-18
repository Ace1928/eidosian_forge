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
def _check_tf2_flags(flags):
    """Checks the parsed and unparsed flags to ensure they are valid in 2.X.

  Args:
    flags: argparse.Namespace object containing TFLite flags.

  Raises:
    ValueError: Invalid flags.
  """
    if not flags.keras_model_file and (not flags.saved_model_dir):
        raise ValueError('one of the arguments --saved_model_dir --keras_model_file is required')