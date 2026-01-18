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
def _create_example_string(example_dict):
    """Create a serialized tf.example from feature dictionary."""
    example = example_pb2.Example()
    for feature_name, feature_list in example_dict.items():
        if not isinstance(feature_list, list):
            raise ValueError('feature value must be a list, but %s: "%s" is %s' % (feature_name, feature_list, type(feature_list)))
        if isinstance(feature_list[0], float):
            example.features.feature[feature_name].float_list.value.extend(feature_list)
        elif isinstance(feature_list[0], str):
            example.features.feature[feature_name].bytes_list.value.extend([f.encode('utf8') for f in feature_list])
        elif isinstance(feature_list[0], bytes):
            example.features.feature[feature_name].bytes_list.value.extend(feature_list)
        elif isinstance(feature_list[0], int):
            example.features.feature[feature_name].int64_list.value.extend(feature_list)
        else:
            raise ValueError('Type %s for value %s is not supported for tf.train.Feature.' % (type(feature_list[0]), feature_list[0]))
    return example.SerializeToString()