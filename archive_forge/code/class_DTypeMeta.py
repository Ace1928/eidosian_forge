import abc
import builtins
import dataclasses
from typing import Type, Sequence, Optional
import numpy as np
from tensorflow.core.framework import types_pb2
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.framework import _dtypes
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.types import doc_typealias
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.types import trace
from tensorflow.core.function import trace_type
from tensorflow.tools.docs import doc_controls
from tensorflow.tsl.python.lib.core import pywrap_ml_dtypes
class DTypeMeta(type(_dtypes.DType), abc.ABCMeta):
    pass