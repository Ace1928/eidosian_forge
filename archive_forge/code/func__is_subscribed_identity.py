import contextlib
import re
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
def _is_subscribed_identity(tensor):
    """Checks if the given tensor is an identity op returned by `subscribe()`.

  Args:
    tensor: A `tf.Tensor` to check.

  Returns:
    True if the given tensor matches the criteria for subscription identities:
    its op type is `Identity`, its name matches the name of its input and
    conforms to the convention for subscribed nodes.
    False otherwise.
  """
    if tensor.op.type != 'Identity':
        return False
    match = re.match('(?P<prefix_name>^.*?)/subscription/Identity[^/]+', tensor.name)
    if match is None or len(match.groups()) != 1:
        return False
    prefix_name = match.group('prefix_name')
    assert len(tensor.op.inputs) == 1, 'Op {} must only have one input'.format(tensor.op.name)
    source_tensor = tensor.op.inputs[0]
    if prefix_name != source_tensor.op.name:
        return False
    return True