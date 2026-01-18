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
def _check_tf1_flags(flags, unparsed):
    """Checks the parsed and unparsed flags to ensure they are valid in 1.X.

  Raises an error if previously support unparsed flags are found. Raises an
  error for parsed flags that don't meet the required conditions.

  Args:
    flags: argparse.Namespace object containing TFLite flags.
    unparsed: List of unparsed flags.

  Raises:
    ValueError: Invalid flags.
  """

    def _get_message_unparsed(flag, orig_flag, new_flag):
        if flag.startswith(orig_flag):
            return '\n  Use {0} instead of {1}'.format(new_flag, orig_flag)
        return ''
    if unparsed:
        output = ''
        for flag in unparsed:
            output += _get_message_unparsed(flag, '--input_file', '--graph_def_file')
            output += _get_message_unparsed(flag, '--savedmodel_directory', '--saved_model_dir')
            output += _get_message_unparsed(flag, '--std_value', '--std_dev_values')
            output += _get_message_unparsed(flag, '--batch_size', '--input_shapes')
            output += _get_message_unparsed(flag, '--dump_graphviz', '--dump_graphviz_dir')
        if output:
            raise ValueError(output)
    if flags.graph_def_file and (not flags.input_arrays or not flags.output_arrays):
        raise ValueError('--input_arrays and --output_arrays are required with --graph_def_file')
    if flags.input_shapes:
        if not flags.input_arrays:
            raise ValueError('--input_shapes must be used with --input_arrays')
        if flags.input_shapes.count(':') != flags.input_arrays.count(','):
            raise ValueError('--input_shapes and --input_arrays must have the same number of items')
    if flags.std_dev_values or flags.mean_values:
        if bool(flags.std_dev_values) != bool(flags.mean_values):
            raise ValueError('--std_dev_values and --mean_values must be used together')
        if flags.std_dev_values.count(',') != flags.mean_values.count(','):
            raise ValueError('--std_dev_values, --mean_values must have the same number of items')
    if (flags.default_ranges_min is None) != (flags.default_ranges_max is None):
        raise ValueError('--default_ranges_min and --default_ranges_max must be used together')
    if flags.dump_graphviz_video and (not flags.dump_graphviz_dir):
        raise ValueError('--dump_graphviz_video must be used with --dump_graphviz_dir')
    if flags.custom_opdefs and (not flags.experimental_new_converter):
        raise ValueError('--custom_opdefs must be used with --experimental_new_converter')
    if flags.custom_opdefs and (not flags.allow_custom_ops):
        raise ValueError('--custom_opdefs must be used with --allow_custom_ops')
    if flags.experimental_select_user_tf_ops and (not flags.experimental_new_converter):
        raise ValueError('--experimental_select_user_tf_ops must be used with --experimental_new_converter')