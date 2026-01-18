from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.tools.docs import doc_controls
class NonNeg(Constraint):
    """Constrains the weights to be non-negative.

  Also available via the shortcut function `tf.keras.constraints.non_neg`.
  """

    def __call__(self, w):
        return w * math_ops.cast(math_ops.greater_equal(w, 0.0), backend.floatx())