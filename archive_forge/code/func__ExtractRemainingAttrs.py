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
def _ExtractRemainingAttrs(op_type_name, op_def, keywords, default_type_attr_map, attrs):
    """Extracts the remaining attributes into `attrs` in _apply_op_helper."""
    for attr in op_def.attr:
        if attr.name in attrs:
            if attr.name in keywords:
                raise TypeError(f"Should not specify value for inferred attr '{attr.name}' for {op_type_name}.")
            continue
        if attr.name in keywords:
            attrs[attr.name] = keywords.pop(attr.name)
        elif attr.name + '_' in keywords:
            attrs[attr.name] = keywords.pop(attr.name + '_')
        elif attr.name in default_type_attr_map:
            attrs[attr.name] = default_type_attr_map[attr.name]
        else:
            raise TypeError(f'No argument found for attr {attr.name} for {op_type_name}')