from tensorflow.python.framework import dtypes
from tensorflow.python.util import object_identity
def _input_index(op, handle):
    """Returns the index of `handle` in `op.inputs`.

  Args:
    op: Operation.
    handle: Resource handle.

  Returns:
    Index in `op.inputs` receiving the resource `handle`.

  Raises:
    ValueError: If handle and its replicated input are both not found in
    `op.inputs`.
  """
    for i, t in enumerate(op.inputs):
        if handle is t:
            return i
    raise ValueError(f'{handle!s} not in list of inputs for op: {op!r}')