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
def _GetInferGraph(self, run_params, saved_model_dir):
    """Return trt converted graphdef."""
    conversion_params = self.GetConversionParams(run_params)
    logging.info(conversion_params)
    converter = self._CreateConverter(run_params, saved_model_dir, conversion_params)
    converter.convert()
    if run_params.is_v2:
        try:
            line_length = max(160, os.get_terminal_size().columns)
        except OSError:
            line_length = 160
        converter.summary(line_length=line_length, detailed=True)
    if run_params.dynamic_shape and self._ShouldConverterBuild(run_params):
        logging.info('Using build mode')

        def _BuildInputFn():
            for shapes in self._GetParamsCached().input_dims:
                yield [array_ops.zeros(x, dtype=spec.dtype) for x, spec in zip(shapes, self._GetParamsCached().input_specs)]
        converter.build(input_fn=_BuildInputFn)
    trt_saved_model_dir = self._GetSavedModelDir(run_params, GraphState.INFERENCE)
    converter.save(trt_saved_model_dir)
    return trt_saved_model_dir