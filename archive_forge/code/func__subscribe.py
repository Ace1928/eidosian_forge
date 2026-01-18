import contextlib
import re
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
def _subscribe(tensor, side_effects, control_cache):
    """Helper method that subscribes a single tensor to a list of side_effects.

  This method will check if the given tensor has already been subscribed or if
  it's a tensor returned by a previous call to `subscribe()` and, if so, will
  reuse the existing identity op, appending the given side effects to the list
  of existing ones.

  Args:
    tensor: The `tf.Tensor` to be subscribed.
    side_effects: List of side_effect functions, see subscribe for details.
    control_cache: `_ControlOutputCache` helper to get control_outputs faster.

  Returns:
    The modified replacement to the passed in tensor which triggers the side
    effects or the given tensor, if it was already been subscribed.
  """
    if not tensor.dtype.is_numpy_compatible:
        logging.debug('Tensor {} has an un-supported {} type and cannot be subscribed.'.format(tensor.name, tensor.dtype))
        return tensor
    if _is_subscribed_identity(tensor):
        return _subscribe_extend(tensor, side_effects)
    name_scope = tensor.op.name + '/subscription/Identity'
    consumers = tensor.consumers()
    matching_ops = [op for op in consumers if op.name.startswith(name_scope)]
    assert len(matching_ops) <= 1, 'Op {} must only have one subscription op connected to it'.format(tensor.op.name)
    if len(matching_ops) == 1:
        candidate_tensor = matching_ops[0].outputs[0]
        if _is_subscribed_identity(candidate_tensor):
            return _subscribe_extend(candidate_tensor, side_effects)
    return _subscribe_new(tensor, side_effects, control_cache)