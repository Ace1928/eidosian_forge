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
def _VerifyGraphDefV1(self, run_params, original_gdef, gdef_to_verify, graph_state):
    expected_engines = self.ExpectedEnginesToBuild(run_params)
    num_engines = 0
    functions = [f.signature.name for f in gdef_to_verify.library.function]
    for node in gdef_to_verify.node:
        if node.op == 'TRTEngineOp':
            logging.info('Found TRTEngineOp: ' + node.name)
            num_engines += 1
            segment_funcdef_name = node.attr['segment_func'].func.name
            function_name = node.name + '_native_segment'
            is_dynamic_engine = not node.attr['static_engine'].b
            self.assertNotEmpty(segment_funcdef_name, node.name)
            self.assertIn(function_name, functions)
            if not IsQuantizationWithCalibration(run_params) and (not is_dynamic_engine):
                self.assertTrue(len(node.attr['serialized_segment'].s), node.name)
            self.assertIn(self._RemoveGraphSequenceNumber(node.name), expected_engines)
            if IsQuantizationWithoutCalibration(run_params):
                if self._ToBytes('INT8') != node.attr['precision_mode'].s:
                    self.assertEqual(self._ToBytes('FP16'), node.attr['precision_mode'].s, node.name)
            else:
                self.assertEqual(self._ToBytes(run_params.precision_mode), node.attr['precision_mode'].s, node.name)
            self.assertEqual(run_params.dynamic_engine, is_dynamic_engine, node.name)
            self.assertEqual(node.attr['use_calibration'].b, run_params.use_calibration, node.name)
            has_calibration_data = len(node.attr['calibration_data'].s)
            if IsQuantizationWithCalibration(run_params) and graph_state == GraphState.INFERENCE:
                self.assertTrue(has_calibration_data, node.name)
            else:
                self.assertFalse(has_calibration_data, node.name)
    if graph_state == GraphState.ORIGINAL:
        self.assertEqual(0, num_engines)
    else:
        self.assertEqual(num_engines, len(expected_engines))
        expected_connections = self.ExpectedConnections(run_params)
        if expected_connections:
            self._VerifyConnections(expected_engines, expected_connections, original_gdef, gdef_to_verify)
        self._VerifyMaxBatchSizeAnnotations(expected_engines=expected_engines, original_gdef=original_gdef, converted_gdef=gdef_to_verify, expected_max_batch_sizes=self.ExpectedMaxBatchSizes(run_params), default_max_batch_size=self.GetMaxBatchSize(run_params))