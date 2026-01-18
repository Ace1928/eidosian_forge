import contextlib
import re
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
def _subscribe_extend(tensor, side_effects):
    """Helper method to extend the list of side_effects for a subscribed tensor.

  Args:
    tensor: A `tf.Tensor` as returned by subscribe().
    side_effects: List of side_effect functions, see subscribe for details.

  Returns:
    The given subscribed tensor (for API consistency).
  """
    assert len(tensor.op.inputs) == 1, 'Op {} must only have one input'.format(tensor.op.name)
    source_tensor = tensor.op.inputs[0]
    outs = []
    name_scope = source_tensor.op.name + '/subscription/'
    with ops.name_scope(name_scope):
        for s in side_effects:
            outs += s(source_tensor)
    out_ops = [out.op if isinstance(out, tensor_lib.Tensor) else out for out in outs]
    tensor.op._add_control_inputs(out_ops)
    return tensor