from google.protobuf import text_format
from tensorflow.core.config import flags
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import op_def_library_pybind
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
def _CheckOpDeprecation(op_type_name, op_def, producer):
    """Checks if the op is deprecated."""
    deprecation_version = op_def.deprecation.version
    if deprecation_version and producer >= deprecation_version:
        raise NotImplementedError(f'Op {op_type_name} is not available in GraphDef version {producer}. It has been removed in version {deprecation_version}. {op_def.deprecation.explanation}.')