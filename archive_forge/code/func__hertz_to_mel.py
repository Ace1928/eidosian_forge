from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import shape_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _hertz_to_mel(frequencies_hertz, name=None):
    """Converts frequencies in `frequencies_hertz` in Hertz to the mel scale.

  Args:
    frequencies_hertz: A `Tensor` of frequencies in Hertz.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of the same shape and type of `frequencies_hertz` containing
    frequencies in the mel scale.
  """
    with ops.name_scope(name, 'hertz_to_mel', [frequencies_hertz]):
        frequencies_hertz = ops.convert_to_tensor(frequencies_hertz)
        return _MEL_HIGH_FREQUENCY_Q * math_ops.log(1.0 + frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ)