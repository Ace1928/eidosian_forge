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
def _Flatten(l):
    """Converts [1, 2, [3, 4], [5]] to [1, 2, 3, 4, 5]."""
    l_of_l = [x if _IsListValue(x) else [x] for x in l]
    return [item for sublist in l_of_l for item in sublist]