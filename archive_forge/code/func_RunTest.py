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
def RunTest(self, run_params):
    with disable_tensorfloat32():
        with trace.Trace(run_params.test_name):
            should_run, reason_for_skipping = self.ShouldRunTest(run_params)
            if not should_run:
                return self.skipTest(reason_for_skipping)
            saved_model_dir = self._MakeSavedModel(run_params)
            np.random.seed(12345)
            inputs_data = []
            input_specs = self._GetParamsCached().input_specs
            for dim_list in self._GetParamsCached().input_dims:
                assert len(input_specs) == len(dim_list), f'Inconsistent input_specs and dim_list: len({input_specs}) != len({dim_list}).'
                current_input_data = []
                for spec, np_shape in zip(input_specs, dim_list):
                    np_dtype = spec.dtype.as_numpy_dtype()
                    if not np.issubdtype(np_dtype, np.bool_):
                        scale = 10.0 if np.issubdtype(np_dtype, np.integer) else 1.0
                        data = (scale * np.random.random_sample(np_shape)).astype(np_dtype)
                    else:
                        data = np.random.choice(a=[False, True], size=np_shape)
                    if run_params.is_v2:
                        with ops.device('/GPU:0'):
                            data = ops.convert_to_tensor(data)
                    current_input_data.append(data)
                inputs_data.append(current_input_data)
            self._VerifyGraphDef(run_params, saved_model_dir, saved_model_dir, GraphState.ORIGINAL)
            logging.info('Running original graph w/o TensorRT\n')
            ref_result = self._RunGraph(run_params, saved_model_dir, inputs_data, GraphState.ORIGINAL, num_runs=1)
            if IsQuantizationWithCalibration(run_params):
                infer_saved_model_dir = self._GetCalibratedInferGraph(run_params, saved_model_dir, inputs_data)
                self._VerifyGraphDef(run_params, saved_model_dir, infer_saved_model_dir, GraphState.INFERENCE)
            elif not run_params.convert_online:
                infer_saved_model_dir = self._GetInferGraph(run_params, saved_model_dir)
                self._VerifyGraphDef(run_params, saved_model_dir, infer_saved_model_dir, GraphState.INFERENCE)
            else:
                infer_saved_model_dir = saved_model_dir
            logging.info('Running final inference graph\n')
            result = self._RunGraph(run_params, infer_saved_model_dir, inputs_data, GraphState.INFERENCE)
            self.assertAllClose(ref_result, result, atol=self.ExpectedAbsoluteTolerance(run_params), rtol=self.ExpectedRelativeTolerance(run_params))