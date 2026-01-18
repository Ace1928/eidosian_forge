import abc
import math
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util import nest
class PiecewiseConstantDecay(LearningRateSchedule):
    """A LearningRateSchedule that uses a piecewise constant decay schedule.

  The function returns a 1-arg callable to compute the piecewise constant
  when passed the current optimizer step. This can be useful for changing the
  learning rate value across different invocations of optimizer functions.

  Example: use a learning rate that's 1.0 for the first 100001 steps, 0.5
    for the next 10000 steps, and 0.1 for any additional steps.

  ```python
  step = tf.Variable(0, trainable=False)
  boundaries = [100000, 110000]
  values = [1.0, 0.5, 0.1]
  learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
      boundaries, values)

  # Later, whenever we perform an optimization step, we pass in the step.
  learning_rate = learning_rate_fn(step)
  ```

  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate. The learning rate schedule is also serializable and
  deserializable using `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.

  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as the boundary tensors.

    The output of the 1-arg function that takes the `step`
    is `values[0]` when `step <= boundaries[0]`,
    `values[1]` when `step > boundaries[0]` and `step <= boundaries[1]`, ...,
    and values[-1] when `step > boundaries[-1]`.
  """

    def __init__(self, boundaries, values, name=None):
        """Piecewise constant from boundaries and interval values.

    Args:
      boundaries: A list of `Tensor`s or `int`s or `float`s with strictly
        increasing entries, and with all elements having the same type as the
        optimizer step.
      values: A list of `Tensor`s or `float`s or `int`s that specifies the
        values for the intervals defined by `boundaries`. It should have one
        more element than `boundaries`, and all elements should have the same
        type.
      name: A string. Optional name of the operation. Defaults to
        'PiecewiseConstant'.

    Raises:
      ValueError: if the number of elements in the lists do not match.
    """
        super(PiecewiseConstantDecay, self).__init__()
        if len(boundaries) != len(values) - 1:
            raise ValueError('The length of boundaries should be 1 less than the length of values')
        self.boundaries = boundaries
        self.values = values
        self.name = name

    def __call__(self, step):
        with ops.name_scope_v2(self.name or 'PiecewiseConstant'):
            boundaries = nest.map_structure(tensor_conversion.convert_to_tensor_v2_with_dispatch, nest.flatten(self.boundaries))
            values = nest.map_structure(tensor_conversion.convert_to_tensor_v2_with_dispatch, nest.flatten(self.values))
            x_recomp = tensor_conversion.convert_to_tensor_v2_with_dispatch(step)
            for i, b in enumerate(boundaries):
                if b.dtype.base_dtype != x_recomp.dtype.base_dtype:
                    b = math_ops.cast(b, x_recomp.dtype.base_dtype)
                    boundaries[i] = b
            pred_fn_pairs = []
            pred_fn_pairs.append((x_recomp <= boundaries[0], lambda: values[0]))
            pred_fn_pairs.append((x_recomp > boundaries[-1], lambda: values[-1]))
            for low, high, v in zip(boundaries[:-1], boundaries[1:], values[1:-1]):
                pred = (x_recomp > low) & (x_recomp <= high)
                pred_fn_pairs.append((pred, lambda v=v: v))
            default = lambda: values[0]
            return control_flow_case.case(pred_fn_pairs, default, exclusive=True)

    def get_config(self):
        return {'boundaries': self.boundaries, 'values': self.values, 'name': self.name}