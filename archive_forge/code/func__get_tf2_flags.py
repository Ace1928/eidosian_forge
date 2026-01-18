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
def _get_tf2_flags(parser):
    """Returns ArgumentParser for tflite_convert for TensorFlow 2.0.

  Args:
    parser: ArgumentParser
  """
    input_file_group = parser.add_mutually_exclusive_group()
    input_file_group.add_argument('--saved_model_dir', type=str, help='Full path of the directory containing the SavedModel.')
    input_file_group.add_argument('--keras_model_file', type=str, help='Full filepath of HDF5 file containing tf.Keras model.')
    parser.add_argument('--saved_model_tag_set', type=str, help='Comma-separated set of tags identifying the MetaGraphDef within the SavedModel to analyze. All tags must be present. In order to pass in an empty tag set, pass in "". (default "serve")')
    parser.add_argument('--saved_model_signature_key', type=str, help='Key identifying the SignatureDef containing inputs and outputs. (default DEFAULT_SERVING_SIGNATURE_DEF_KEY)')
    parser.add_argument('--enable_v1_converter', action='store_true', help='Enables the TensorFlow V1 converter in 2.0')