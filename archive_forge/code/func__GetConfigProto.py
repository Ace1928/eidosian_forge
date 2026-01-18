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
def _GetConfigProto(self, run_params, graph_state):
    """Get config proto based on specific settings."""
    conversion_params = self.GetConversionParams(run_params)
    max_batch_size = self.GetMaxBatchSize(run_params)
    if graph_state == GraphState.INFERENCE and run_params.convert_online:
        rewriter_cfg = trt_convert.get_tensorrt_rewriter_config(conversion_params, is_dynamic_op=run_params.dynamic_engine, max_batch_size=max_batch_size, disable_non_trt_optimizers=self._disable_non_trt_optimizers)
    else:
        rewriter_cfg = rewriter_config_pb2.RewriterConfig()
        if self._disable_non_trt_optimizers:
            trt_utils.disable_non_trt_optimizers_in_rewriter_config(rewriter_cfg)
    config = config_pb2.ConfigProto(gpu_options=self._GetGPUOptions(), graph_options=config_pb2.GraphOptions(rewrite_options=rewriter_cfg))
    return config