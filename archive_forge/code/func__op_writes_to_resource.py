from tensorflow.python.framework import dtypes
from tensorflow.python.util import object_identity
def _op_writes_to_resource(handle, op):
    """Returns whether op writes to resource handle.

  Args:
    handle: Resource handle. Must be an input of `op`.
    op: Operation.

  Returns:
    Returns False if op is a read-only op registered using
    `register_read_only_resource_op` or if `handle` is an input at one of
    the indices in the `READ_ONLY_RESOURCE_INPUTS_ATTR` attr of the op, True
    otherwise.

  Raises:
    ValueError: if `handle` is not an input of `op`.
  """
    if op.type in RESOURCE_READ_OPS:
        return False
    input_index = _input_index(op, handle)
    try:
        read_only_input_indices = op.get_attr(READ_ONLY_RESOURCE_INPUTS_ATTR)
    except ValueError:
        return True
    return input_index not in read_only_input_indices