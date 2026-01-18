import distutils.spawn
import enum
import hashlib
import os as _os
import platform as _platform
import subprocess as _subprocess
import tempfile as _tempfile
from typing import Optional
import warnings
from tensorflow.compiler.mlir.quantization.stablehlo import quantization_options_pb2 as quant_opts_pb2
from tensorflow.lite.python import lite_constants
from tensorflow.lite.python import util
from tensorflow.lite.python import wrap_toco
from tensorflow.lite.python.convert_phase import Component
from tensorflow.lite.python.convert_phase import convert_phase
from tensorflow.lite.python.convert_phase import ConverterError
from tensorflow.lite.python.convert_phase import SubComponent
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.lite.python.metrics.wrapper import metrics_wrapper as _metrics_wrapper
from tensorflow.lite.toco import model_flags_pb2 as _model_flags_pb2
from tensorflow.lite.toco import toco_flags_pb2 as _conversion_flags_pb2
from tensorflow.lite.toco import types_pb2 as _types_pb2
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import resource_loader as _resource_loader
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export as _tf_export
def _try_convert_to_unicode(output):
    if output is None:
        return ''
    if isinstance(output, bytes):
        try:
            return output.decode('utf-8')
        except UnicodeDecodeError:
            pass
    return output