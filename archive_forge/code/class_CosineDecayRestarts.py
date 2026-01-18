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
class CosineDecayRestarts(LearningRateSchedule):
    """A LearningRateSchedule that uses a cosine decay schedule with restarts.

  See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
  SGDR: Stochastic Gradient Descent with Warm Restarts.

  When training a model, it is often useful to lower the learning rate as
  the training progresses. This schedule applies a cosine decay function with
  restarts to an optimizer step, given a provided initial learning rate.
  It requires a `step` value to compute the decayed learning rate. You can
  just pass a TensorFlow variable that you increment at each training step.

  The schedule a 1-arg callable that produces a decayed learning
  rate when passed the current optimizer step. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.

  The learning rate multiplier first decays
  from 1 to `alpha` for `first_decay_steps` steps. Then, a warm
  restart is performed. Each new warm restart runs for `t_mul` times more
  steps and with `m_mul` times smaller initial learning rate.

  Example usage:
  ```python
  first_decay_steps = 1000
  lr_decayed_fn = (
    tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate,
        first_decay_steps))
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

    def __init__(self, initial_learning_rate, first_decay_steps, t_mul=2.0, m_mul=1.0, alpha=0.0, name=None):
        """Applies cosine decay with restarts to the learning rate.

    Args:
      initial_learning_rate: A scalar `float32` or `float64` Tensor or a Python
        number. The initial learning rate.
      first_decay_steps: A scalar `int32` or `int64` `Tensor` or a Python
        number. Number of steps to decay over.
      t_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
        Used to derive the number of iterations in the i-th period
      m_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
        Used to derive the initial learning rate of the i-th period:
      alpha: A scalar `float32` or `float64` Tensor or a Python number.
        Minimum learning rate value as a fraction of the initial_learning_rate.
      name: String. Optional name of the operation.  Defaults to 'SGDRDecay'.
    """
        super(CosineDecayRestarts, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.first_decay_steps = first_decay_steps
        self._t_mul = t_mul
        self._m_mul = m_mul
        self.alpha = alpha
        self.name = name

    def __call__(self, step):
        with ops.name_scope_v2(self.name or 'SGDRDecay') as name:
            initial_learning_rate = tensor_conversion.convert_to_tensor_v2_with_dispatch(self.initial_learning_rate, name='initial_learning_rate')
            dtype = initial_learning_rate.dtype
            first_decay_steps = math_ops.cast(self.first_decay_steps, dtype)
            alpha = math_ops.cast(self.alpha, dtype)
            t_mul = math_ops.cast(self._t_mul, dtype)
            m_mul = math_ops.cast(self._m_mul, dtype)
            global_step_recomp = math_ops.cast(step, dtype)
            completed_fraction = global_step_recomp / first_decay_steps

            def compute_step(completed_fraction, geometric=False):
                """Helper for `cond` operation."""
                if geometric:
                    i_restart = math_ops.floor(math_ops.log(1.0 - completed_fraction * (1.0 - t_mul)) / math_ops.log(t_mul))
                    sum_r = (1.0 - t_mul ** i_restart) / (1.0 - t_mul)
                    completed_fraction = (completed_fraction - sum_r) / t_mul ** i_restart
                else:
                    i_restart = math_ops.floor(completed_fraction)
                    completed_fraction -= i_restart
                return (i_restart, completed_fraction)
            i_restart, completed_fraction = cond.cond(math_ops.equal(t_mul, 1.0), lambda: compute_step(completed_fraction, geometric=False), lambda: compute_step(completed_fraction, geometric=True))
            m_fac = m_mul ** i_restart
            cosine_decayed = 0.5 * m_fac * (1.0 + math_ops.cos(constant_op.constant(math.pi) * completed_fraction))
            decayed = (1 - alpha) * cosine_decayed + alpha
            return math_ops.multiply(initial_learning_rate, decayed, name=name)

    def get_config(self):
        return {'initial_learning_rate': self.initial_learning_rate, 'first_decay_steps': self.first_decay_steps, 't_mul': self._t_mul, 'm_mul': self._m_mul, 'alpha': self.alpha, 'name': self.name}