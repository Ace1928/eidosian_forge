import functools
import numpy as np
from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import cond
from tensorflow.python.util import nest
def _verify_spec_compatible(input_name, spec_name, input_, spec):
    """Verifies that a symbol has a type compatible vith a given spec.

  Here, compatibility is viewed in the general TensorFlow sense: that the dtypes
  are the same after implicit conversion, if both are tensors.

  This verifier ensures consistent treatment of types across AutoGraph.

  Args:
    input_name: A name to use for `input_` in error messages.
    spec_name: A name to use for `spec` in error messages.
    input_: Any, value to verify.
    spec: TypeSpec that `input_` must be compatible with.

  Raises:
    ValueError if the two types have been determined not to be compatible.
  """
    assert isinstance(spec, tensor_spec.TensorSpec)
    if input is None:
        raise ValueError('{} cannot be None'.format(input_name))
    if isinstance(input_, (bool, int, float, str, np.ndarray)):
        input_ = tensor_conversion.convert_to_tensor_v2(input_)
    input_dtype = getattr(input_, 'dtype', None)
    if input_dtype != spec.dtype:
        input_dtype_str = 'no dtype' if input_dtype is None else str(input_dtype)
        raise TypeError('{} must have the same dtype as {}. Expected {}, got {}'.format(input_name, spec_name, spec.dtype, input_dtype_str))