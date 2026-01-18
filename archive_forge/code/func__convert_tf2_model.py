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
def _convert_tf2_model(flags):
    """Calls function to convert the TensorFlow 2.0 model into a TFLite model.

  Args:
    flags: argparse.Namespace object.

  Raises:
    ValueError: Unsupported file format.
  """
    if flags.saved_model_dir:
        converter = lite.TFLiteConverterV2.from_saved_model(flags.saved_model_dir, signature_keys=_parse_array(flags.saved_model_signature_key), tags=_parse_set(flags.saved_model_tag_set))
    elif flags.keras_model_file:
        model = keras_deps.get_load_model_function()(flags.keras_model_file)
        converter = lite.TFLiteConverterV2.from_keras_model(model)
    converter.experimental_new_converter = flags.experimental_new_converter
    if flags.experimental_new_quantizer is not None:
        converter.experimental_new_quantizer = flags.experimental_new_quantizer
    tflite_model = converter.convert()
    with gfile.GFile(flags.output_file, 'wb') as f:
        f.write(tflite_model)