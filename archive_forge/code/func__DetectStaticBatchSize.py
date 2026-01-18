import collections
import errno
import gc
import itertools
import os
import re
import shutil
import tempfile
import warnings
from contextlib import contextmanager
import numpy as np
from tensorflow.compiler.tf2tensorrt._pywrap_py_utils import is_tensorrt_enabled
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.compiler.tensorrt import trt_convert
from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest
def _DetectStaticBatchSize(node_def):
    """Returns the static batch size of an operation or None.

      It is incorrect to use the output shapes to find the batch size of an
      operation, as the segmenter actually uses the input shapes. However, it is
      a simplication and works for most of the cases for the test purposes.

      Args:
        node_def: `tf.NodeDef`. The target node for analysis.

      Returns:
        If all the outputs of the node have the same static batch size, returns
        the int value for the batch size. Otherwise returns None.
      """
    shapes = node_def.attr['_output_shapes'].list.shape
    batch_size = set((list(s.dim)[0].size if len(s.dim) >= 2 else None for s in shapes))
    if len(batch_size) == 1 and list(batch_size)[0] >= 1:
        return list(batch_size)[0]
    return None