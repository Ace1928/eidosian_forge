import collections
from functools import partial  # pylint: disable=g-importing-member
import os
import platform
import sys
import tempfile
import numpy as np
import six as _six
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import resource
from tensorflow.python.training import saver
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.tensorrt.ConversionParams', v1=[])
class TrtConversionParams(collections.namedtuple('TrtConversionParams', ['max_workspace_size_bytes', 'precision_mode', 'minimum_segment_size', 'maximum_cached_engines', 'use_calibration', 'allow_build_at_runtime'])):
    """Parameters that are used for TF-TRT conversion.

  Fields:
    max_workspace_size_bytes: the maximum GPU temporary memory that the TRT
      engine can use at execution time. This corresponds to the
      'workspaceSize' parameter of nvinfer1::IBuilder::setMaxWorkspaceSize().
    precision_mode: one of the strings in
      TrtPrecisionMode.supported_precision_modes().
    minimum_segment_size: the minimum number of nodes required for a subgraph
      to be replaced by TRTEngineOp.
    maximum_cached_engines: max number of cached TRT engines for dynamic TRT
      ops. Created TRT engines for a dynamic dimension are cached. If the
      number of cached engines is already at max but none of them supports the
      input shapes, the TRTEngineOp will fall back to run the original TF
      subgraph that corresponds to the TRTEngineOp.
    use_calibration: this argument is ignored if precision_mode is not INT8.
      If set to True, a calibration graph will be created to calibrate the
      missing ranges. The calibration graph must be converted to an inference
      graph by running calibration with calibrate(). If set to False,
      quantization nodes will be expected for every tensor in the graph
      (excluding those which will be fused). If a range is missing, an error
      will occur. Please note that accuracy may be negatively affected if
      there is a mismatch between which tensors TRT quantizes and which
      tensors were trained with fake quantization.
    allow_build_at_runtime: whether to allow building TensorRT engines during
      runtime if no prebuilt TensorRT engine can be found that can handle the
      given inputs during runtime, then a new TensorRT engine is built at
      runtime if allow_build_at_runtime=True, and otherwise native TF is used.
  """

    def __new__(cls, max_workspace_size_bytes=DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES, precision_mode=TrtPrecisionMode.FP32, minimum_segment_size=3, maximum_cached_engines=1, use_calibration=True, allow_build_at_runtime=True):
        return super(TrtConversionParams, cls).__new__(cls, max_workspace_size_bytes, precision_mode, minimum_segment_size, maximum_cached_engines, use_calibration, allow_build_at_runtime)