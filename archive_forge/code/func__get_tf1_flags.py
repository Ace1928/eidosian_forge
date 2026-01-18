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
def _get_tf1_flags(parser):
    """Returns ArgumentParser for tflite_convert for TensorFlow 1.X.

  Args:
    parser: ArgumentParser
  """
    input_file_group = parser.add_mutually_exclusive_group(required=True)
    input_file_group.add_argument('--graph_def_file', type=str, help='Full filepath of file containing frozen TensorFlow GraphDef.')
    input_file_group.add_argument('--saved_model_dir', type=str, help='Full filepath of directory containing the SavedModel.')
    input_file_group.add_argument('--keras_model_file', type=str, help='Full filepath of HDF5 file containing tf.Keras model.')
    parser.add_argument('--output_format', type=str.upper, choices=['TFLITE', 'GRAPHVIZ_DOT'], help='Output file format.')
    parser.add_argument('--inference_type', type=str.upper, default='FLOAT', help='Target data type of real-number arrays in the output file. Must be either FLOAT, INT8 or UINT8.')
    parser.add_argument('--inference_input_type', type=str.upper, help='Target data type of real-number input arrays. Allows for a different type for input arrays in the case of quantization. Must be either FLOAT, INT8 or UINT8.')
    parser.add_argument('--input_arrays', type=str, help='Names of the input arrays, comma-separated.')
    parser.add_argument('--input_shapes', type=str, help='Shapes corresponding to --input_arrays, colon-separated.')
    parser.add_argument('--output_arrays', type=str, help='Names of the output arrays, comma-separated.')
    parser.add_argument('--saved_model_tag_set', type=str, help='Comma-separated set of tags identifying the MetaGraphDef within the SavedModel to analyze. All tags must be present. In order to pass in an empty tag set, pass in "". (default "serve")')
    parser.add_argument('--saved_model_signature_key', type=str, help='Key identifying the SignatureDef containing inputs and outputs. (default DEFAULT_SERVING_SIGNATURE_DEF_KEY)')
    parser.add_argument('--std_dev_values', type=str, help='Standard deviation of training data for each input tensor, comma-separated floats. Used for quantized input tensors. (default None)')
    parser.add_argument('--mean_values', type=str, help='Mean of training data for each input tensor, comma-separated floats. Used for quantized input tensors. (default None)')
    parser.add_argument('--default_ranges_min', type=float, help='Default value for min bound of min/max range values used for all arrays without a specified range, Intended for experimenting with quantization via "dummy quantization". (default None)')
    parser.add_argument('--default_ranges_max', type=float, help='Default value for max bound of min/max range values used for all arrays without a specified range, Intended for experimenting with quantization via "dummy quantization". (default None)')
    parser.add_argument('--quantize_weights', dest='post_training_quantize', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--post_training_quantize', dest='post_training_quantize', action='store_true', help='Boolean indicating whether to quantize the weights of the converted float model. Model size will be reduced and there will be latency improvements (at the cost of accuracy). (default False)')
    parser.add_argument('--quantize_to_float16', dest='quantize_to_float16', action='store_true', help='Boolean indicating whether to quantize weights to fp16 instead of the default int8 when post-training quantization (--post_training_quantize) is enabled. (default False)')
    parser.add_argument('--drop_control_dependency', action='store_true', help='Boolean indicating whether to drop control dependencies silently. This is due to TensorFlow not supporting control dependencies. (default True)')
    parser.add_argument('--reorder_across_fake_quant', action='store_true', help='Boolean indicating whether to reorder FakeQuant nodes in unexpected locations. Used when the location of the FakeQuant nodes is preventing graph transformations necessary to convert the graph. Results in a graph that differs from the quantized training graph, potentially causing differing arithmetic behavior. (default False)')
    parser.add_argument('--change_concat_input_ranges', type=str.upper, choices=['TRUE', 'FALSE'], help='Boolean to change behavior of min/max ranges for inputs and outputs of the concat operator for quantized models. Changes the ranges of concat operator overlap when true. (default False)')
    parser.add_argument('--allow_custom_ops', action=_ParseBooleanFlag, nargs='?', help='Boolean indicating whether to allow custom operations. When false any unknown operation is an error. When true, custom ops are created for any op that is unknown. The developer will need to provide these to the TensorFlow Lite runtime with a custom resolver. (default False)')
    parser.add_argument('--custom_opdefs', type=str, help='String representing a list of custom ops OpDefs delineated with commas that are included in the GraphDef. Required when using custom operations with --experimental_new_converter.')
    parser.add_argument('--target_ops', type=str, help='Experimental flag, subject to change. Set of OpsSet options indicating which converter to use. Options: {0}. One or more option may be specified. (default set([OpsSet.TFLITE_BUILTINS]))'.format(','.join(lite.OpsSet.get_options())))
    parser.add_argument('--experimental_select_user_tf_ops', type=str, help="Experimental flag, subject to change. Comma separated list of user's defined TensorFlow operators required in the runtime.")
    parser.add_argument('--dump_graphviz_dir', type=str, help='Full filepath of folder to dump the graphs at various stages of processing GraphViz .dot files. Preferred over --output_format=GRAPHVIZ_DOT in order to keep the requirements of the output file.')
    parser.add_argument('--dump_graphviz_video', action='store_true', help='Boolean indicating whether to dump the graph after every graph transformation')
    parser.add_argument('--conversion_summary_dir', type=str, help='Full filepath to store the conversion logs, which includes graphviz of the model before/after the conversion, an HTML report and the conversion proto buffers. This will only be generated when passing --experimental_new_converter')