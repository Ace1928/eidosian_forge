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
def _get_tflite_converter(flags):
    """Makes a TFLiteConverter object based on the flags provided.

  Args:
    flags: argparse.Namespace object containing TFLite flags.

  Returns:
    TFLiteConverter object.

  Raises:
    ValueError: Invalid flags.
  """
    input_arrays = _parse_array(flags.input_arrays)
    input_shapes = None
    if flags.input_shapes:
        input_shapes_list = [_parse_array(shape, type_fn=int) for shape in flags.input_shapes.split(':')]
        input_shapes = dict(list(zip(input_arrays, input_shapes_list)))
    output_arrays = _parse_array(flags.output_arrays)
    converter_kwargs = {'input_arrays': input_arrays, 'input_shapes': input_shapes, 'output_arrays': output_arrays}
    if flags.graph_def_file:
        converter_fn = lite.TFLiteConverter.from_frozen_graph
        converter_kwargs['graph_def_file'] = flags.graph_def_file
    elif flags.saved_model_dir:
        converter_fn = lite.TFLiteConverter.from_saved_model
        converter_kwargs['saved_model_dir'] = flags.saved_model_dir
        converter_kwargs['tag_set'] = _parse_set(flags.saved_model_tag_set)
        converter_kwargs['signature_key'] = flags.saved_model_signature_key
    elif flags.keras_model_file:
        converter_fn = lite.TFLiteConverter.from_keras_model_file
        converter_kwargs['model_file'] = flags.keras_model_file
    else:
        raise ValueError('--graph_def_file, --saved_model_dir, or --keras_model_file must be specified.')
    return converter_fn(**converter_kwargs)