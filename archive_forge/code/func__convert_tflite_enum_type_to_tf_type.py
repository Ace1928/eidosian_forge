import copy
import datetime
import sys
from absl import logging
import flatbuffers
from tensorflow.core.protobuf import config_pb2 as _config_pb2
from tensorflow.core.protobuf import meta_graph_pb2 as _meta_graph_pb2
from tensorflow.lite.python import conversion_metadata_schema_py_generated as conversion_metadata_fb
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.lite.python import tflite_keras_util as _tflite_keras_util
from tensorflow.lite.python.op_hint import convert_op_hints_to_stubs
from tensorflow.lite.python.op_hint import find_all_hinted_output_nodes
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.python.eager import function
from tensorflow.python.framework import convert_to_constants as _convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import error_interpolation as _error_interpolation
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training.saver import export_meta_graph as _export_meta_graph
def _convert_tflite_enum_type_to_tf_type(tflite_enum_type):
    """Converts tflite enum type (eg: 0) to tf type (eg: tf.float32).

  Args:
    tflite_enum_type: tflite enum type (eg: 0, that corresponds to float32)

  Raises:
    ValueError: If an invalid tflite enum type is provided.

  Returns:
    tf type (eg: tf.float32)
  """
    tf_type = _MAP_TFLITE_ENUM_TO_TF_TYPES.get(tflite_enum_type)
    if tf_type is None:
        raise ValueError('Unsupported enum {}. The valid map of enum to tf types is : {}'.format(tflite_enum_type, _MAP_TFLITE_ENUM_TO_TF_TYPES))
    return tf_type