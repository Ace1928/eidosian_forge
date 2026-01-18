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
def _ExtractDefaultTypesAndAllowedTypes(op_def, default_type_attr_map, allowed_list_attr_map):
    """Extracts the `default_type_attr_map` and `allowed_list_attr_map`."""
    for attr_def in op_def.attr:
        if attr_def.type != 'type':
            continue
        key = attr_def.name
        if attr_def.HasField('default_value'):
            default_type_attr_map[key] = dtypes.as_dtype(attr_def.default_value.type)
        if attr_def.HasField('allowed_values'):
            allowed_list_attr_map[key] = attr_def.allowed_values.list.type