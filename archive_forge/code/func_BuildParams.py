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
def BuildParams(self, graph_fn, dtype, input_shapes, output_shapes):
    """Build test parameters.

    The input_shapes and output_shapes arguments are known (static) shapes that
    can be used to generate test data. To define the model, we also specify
    corresponding input/output TensorSpecs. These are defined using the shape
    arguments. For each input tensor we define:

    input_spec = [None] + input_shape[1:]

    and similarly for output shapes. This means that we leave the first (batch)
    dimension unknown, the rest is just copied from the shapes arg.

    Args:
      graph_fn: The function to build the graph.
      dtype: The element type.
      input_shapes: The input shapes.
      output_shapes: The output shapes.

    Returns:
      The test parameters.
    """
    input_mask = [[False] + [True] * (len(shape) - 1) for shape in input_shapes]
    output_mask = [[False] + [True] * (len(shape) - 1) if shape else [] for shape in output_shapes]
    return self.BuildParamsWithMask(graph_fn, dtype, input_shapes, output_shapes, input_mask, output_mask, [], [])