from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.tools.docs import doc_controls
class UnitNorm(Constraint):
    """Constrains the weights incident to each hidden unit to have unit norm.

  Also available via the shortcut function `tf.keras.constraints.unit_norm`.

  Args:
    axis: integer, axis along which to calculate weight norms.
      For instance, in a `Dense` layer the weight matrix
      has shape `(input_dim, output_dim)`,
      set `axis` to `0` to constrain each weight vector
      of length `(input_dim,)`.
      In a `Conv2D` layer with `data_format="channels_last"`,
      the weight tensor has shape
      `(rows, cols, input_depth, output_depth)`,
      set `axis` to `[0, 1, 2]`
      to constrain the weights of each filter tensor of size
      `(rows, cols, input_depth)`.
  """

    def __init__(self, axis=0):
        self.axis = axis

    @doc_controls.do_not_generate_docs
    def __call__(self, w):
        return w / (backend.epsilon() + backend.sqrt(math_ops.reduce_sum(math_ops.square(w), axis=self.axis, keepdims=True)))

    @doc_controls.do_not_generate_docs
    def get_config(self):
        return {'axis': self.axis}