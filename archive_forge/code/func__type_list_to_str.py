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
def _type_list_to_str(types):
    if any((_ not in _DTYPE_TO_STR for _ in types)):
        unsupported_types = [type_ for type_ in types if type_ not in _DTYPE_TO_STR]
        raise ValueError(f'Unsupported dtypes {unsupported_types} in `types`. Supported dtypes are {_DTYPE_TO_STR.keys()}.')
    return ''.join((_DTYPE_TO_STR[_] for _ in types))