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
class LinearCosineDecay(LearningRateSchedule):
    """A LearningRateSchedule that uses a linear cosine decay schedule.

  See [Bello et al., ICML2017] Neural Optimizer Search with RL.
  https://arxiv.org/abs/1709.07417

  For the idea of warm starts here controlled by `num_periods`,
  see [Loshchilov & Hutter, ICLR2016] SGDR: Stochastic Gradient Descent
  with Warm Restarts. https://arxiv.org/abs/1608.03983

  Note that linear cosine decay is more aggressive than cosine decay and
  larger initial learning rates can typically be used.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses. This schedule applies a linear cosine decay
  function to an optimizer step, given a provided initial learning rate.
  It requires a `step` value to compute the decayed learning rate. You can
  just pass a TensorFlow variable that you increment at each training step.

  The schedule a 1-arg callable that produces a decayed learning
  rate when passed the current optimizer step. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  It is computed as:

  ```python
  def decayed_learning_rate(step):
    step = min(step, decay_steps)
    linear_decay = (decay_steps - step) / decay_steps
    cosine_decay = 0.5 * (
        1 + cos(pi * 2 * num_periods * step / decay_steps))
    decayed = (alpha + linear_decay) * cosine_decay + beta
    return initial_learning_rate * decayed
  ```

  Example usage:
  ```python
  decay_steps = 1000
  lr_decayed_fn = (
    tf.keras.experimental.LinearCosineDecay(
      initial_learning_rate, decay_steps))
  ```

  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate. The learning rate schedule is also serializable and
  deserializable using `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.

  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
  """

    def __init__(self, initial_learning_rate, decay_steps, num_periods=0.5, alpha=0.0, beta=0.001, name=None):
        """Applies linear cosine decay to the learning rate.

    Args:
      initial_learning_rate: A scalar `float32` or `float64` Tensor or a Python
        number. The initial learning rate.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Number of steps to decay over.
      num_periods: Number of periods in the cosine part of the decay.
        See computation above.
      alpha: See computation above.
      beta: See computation above.
      name: String.  Optional name of the operation.  Defaults to
        'LinearCosineDecay'.
    """
        super(LinearCosineDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.num_periods = num_periods
        self.alpha = alpha
        self.beta = beta
        self.name = name

    def __call__(self, step):
        with ops.name_scope_v2(self.name or 'LinearCosineDecay') as name:
            initial_learning_rate = tensor_conversion.convert_to_tensor_v2_with_dispatch(self.initial_learning_rate, name='initial_learning_rate')
            dtype = initial_learning_rate.dtype
            decay_steps = math_ops.cast(self.decay_steps, dtype)
            num_periods = math_ops.cast(self.num_periods, dtype)
            alpha = math_ops.cast(self.alpha, dtype)
            beta = math_ops.cast(self.beta, dtype)
            global_step_recomp = math_ops.cast(step, dtype)
            global_step_recomp = math_ops.minimum(global_step_recomp, decay_steps)
            linear_decayed = (decay_steps - global_step_recomp) / decay_steps
            completed_fraction = global_step_recomp / decay_steps
            fraction = 2.0 * num_periods * completed_fraction
            cosine_decayed = 0.5 * (1.0 + math_ops.cos(constant_op.constant(math.pi) * fraction))
            linear_cosine_decayed = (alpha + linear_decayed) * cosine_decayed + beta
            return math_ops.multiply(initial_learning_rate, linear_cosine_decayed, name=name)

    def get_config(self):
        return {'initial_learning_rate': self.initial_learning_rate, 'decay_steps': self.decay_steps, 'num_periods': self.num_periods, 'alpha': self.alpha, 'beta': self.beta, 'name': self.name}