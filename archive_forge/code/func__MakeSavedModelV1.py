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
def _MakeSavedModelV1(self, run_params):
    """Write the saved model as an input for testing."""
    params = self._GetParamsCached()
    g = ops.Graph()
    with g.as_default():
        inputs = []
        for spec in params.input_specs:
            inp = array_ops.placeholder(dtype=spec.dtype, shape=spec.shape, name=spec.name)
            inputs.append(inp)
        outputs = params.graph_fn(*inputs)
        if not isinstance(outputs, list) and (not isinstance(outputs, tuple)):
            outputs = [outputs]
    signature_def = signature_def_utils.build_signature_def(inputs={inp.op.name: utils.build_tensor_info(inp) for inp in inputs}, outputs={out.op.name: utils.build_tensor_info(out) for out in outputs}, method_name=signature_constants.PREDICT_METHOD_NAME)
    saved_model_dir = self._GetSavedModelDir(run_params, GraphState.ORIGINAL)
    saved_model_builder = builder.SavedModelBuilder(saved_model_dir)
    with self.session(graph=g, config=self._GetConfigProto(run_params, GraphState.ORIGINAL)) as sess:
        saved_model_builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], signature_def_map={signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def})
    saved_model_builder.save()
    return saved_model_dir