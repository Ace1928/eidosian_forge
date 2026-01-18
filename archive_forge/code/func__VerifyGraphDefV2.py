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
def _VerifyGraphDefV2(self, run_params, original_gdef, gdef_to_verify, graph_state):
    if graph_state == GraphState.ORIGINAL:
        return
    expected_engines = self.ExpectedEnginesToBuild(run_params)
    all_op_names = [node.name for node in gdef_to_verify.node]
    trt_op_names = []
    for func in gdef_to_verify.library.function:
        if not re.search('TRTEngineOp_\\d{3,}_\\d{3,}_native_segment', func.signature.name):
            for node in func.node_def:
                all_op_names.append(node.name)
                if node.op == 'TRTEngineOp':
                    trt_op_names.append(node.name)
                    if run_params.dynamic_shape:
                        self.assertEqual(self._ToString(node.attr['profile_strategy'].s).lower(), self._profile_strategy.lower())
    all_op_names = self._Canonicalize(all_op_names)
    trt_op_names = self._RemoveGraphSequenceNumber(self._Canonicalize(trt_op_names))
    if isinstance(expected_engines, dict):
        unexpected_names = set(nest.flatten(expected_engines.values()))
        self.assertEmpty([name for name in unexpected_names if name in all_op_names])
        expected_engines = set(expected_engines.keys())
    self.assertEqual(set(expected_engines), trt_op_names)