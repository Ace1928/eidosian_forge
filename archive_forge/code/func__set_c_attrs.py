import collections
import hashlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import context
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
def _set_c_attrs(self, attrs):
    """Sets `attrs` as attributes of self._c_func.

    Requires that self._c_func is not None.

    Args:
      attrs: a dictionary from attribute name to attribute proto value
    """
    for name, attr_value in attrs.items():
        serialized = attr_value.SerializeToString()
        with self._c_func.get() as func:
            c_api.TF_FunctionSetAttrValueProto(func, compat.as_str(name), serialized)