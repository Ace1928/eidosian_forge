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
def _RunGraphV1(self, saved_model_dir, inputs_data, config, num_runs=2):
    """Run given graphdef multiple times using TF 1.x runtime."""
    params = self._GetParamsCached()
    fetches = self._GetFetchNames()
    g = ops.Graph()
    with g.as_default():
        with self.session(graph=g, config=config, use_gpu=True) as sess:
            loader.load(sess, [tag_constants.SERVING], saved_model_dir)
            vals = []
            for expected_shapes, current_input_data in zip(params.expected_output_dims, inputs_data):
                val = None
                for _ in range(num_runs):
                    new_val = sess.run(fetches, self._GetFeedDict(current_input_data))
                    self.assertEqual(len(expected_shapes), len(new_val))
                    for expected_shape, actual_val in zip(expected_shapes, new_val):
                        self.assertEqual(list(expected_shape), list(actual_val.shape))
                    if val is not None:
                        self.assertAllClose(val, new_val, atol=1e-05, rtol=1e-05)
                    val = new_val
                vals.append(val)
            return vals