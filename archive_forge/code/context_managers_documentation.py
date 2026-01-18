import contextlib
from tensorflow.python.framework import ops
from tensorflow.python.ops import tensor_array_ops
Create a TF control dependency on the return values of a function.

  If the function had no return value, a no-op context is returned.

  Args:
    return_value: The return value to set as control dependency.

  Returns:
    A context manager.
  