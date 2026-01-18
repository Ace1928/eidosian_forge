import contextlib
from tensorflow.core.framework import graph_pb2
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import control_flow_util
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def _ConvertInputMapValues(name, input_map):
    """Ensures all input map values are tensors.

  This should be called from inside the import name scope.

  Args:
    name: the `name` argument passed to import_graph_def
    input_map: the `input_map` argument passed to import_graph_def.

  Returns:
    An possibly-updated version of `input_map`.

  Raises:
    ValueError: if input map values cannot be converted due to empty name scope.
  """
    if not all((isinstance(v, tensor.Tensor) for v in input_map.values())):
        if name == '':
            raise ValueError('tf.import_graph_def() requires a non-empty `name` if `input_map` contains non-Tensor values. Try calling tf.convert_to_tensor() on `input_map` values before calling tf.import_graph_def().')
        with ops.name_scope('_inputs'):
            input_map = {k: ops.convert_to_tensor(v) for k, v in input_map.items()}
    return input_map