from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import shape_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _mel_to_hertz(mel_values, name=None):
    """Converts frequencies in `mel_values` from the mel scale to linear scale.

  Args:
    mel_values: A `Tensor` of frequencies in the mel scale.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of the same shape and type as `mel_values` containing linear
    scale frequencies in Hertz.
  """
    with ops.name_scope(name, 'mel_to_hertz', [mel_values]):
        mel_values = ops.convert_to_tensor(mel_values)
        return _MEL_BREAK_FREQUENCY_HERTZ * (math_ops.exp(mel_values / _MEL_HIGH_FREQUENCY_Q) - 1.0)