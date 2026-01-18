import contextlib
from tensorflow.compiler.jit.ops import xla_ops
from tensorflow.compiler.jit.ops import xla_ops_grad  # pylint: disable=unused-import
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import summary_op_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def _postprocess_flat_outputs(outputs):
    """Validates flat outputs and adds back device assignments.

  Args:
    outputs: Output from `computation` inside `xla.compile`.

  Returns:
    Tensors and Operations extracted from outputs.
  """
    if outputs is None:
        outputs = tuple()
    if not isinstance(outputs, collections_abc.Sequence):
        outputs = (outputs,)
    outputs += (control_flow_ops.no_op(),)
    try:
        outputs = [o if isinstance(o, ops.Operation) else ops.convert_to_tensor(o) for o in outputs]
    except Exception as e:
        raise ValueError('XLA computation function return values must all either be Operations or convertible to Tensors. Got error: "%s"' % str(e))
    output_operations = [o for o in outputs if isinstance(o, ops.Operation)]
    output_tensors = [o for o in outputs if not isinstance(o, ops.Operation)]
    if outputs != output_tensors + output_operations:
        raise ValueError('XLA computation function must return zero or more Tensor values followed by zero or more Operations.')
    new_output_tensors = []
    for t in output_tensors:
        with ops.device(t.device if t.device else ''):
            new_output_tensors.append(array_ops.identity(t))
    return (new_output_tensors, output_operations)