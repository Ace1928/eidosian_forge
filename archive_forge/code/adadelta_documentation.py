import numpy as np
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.training import gen_training_ops
Optimizer that implements the Adadelta algorithm.

  Adadelta optimization is a stochastic gradient descent method that is based on
  adaptive learning rate per dimension to address two drawbacks:

  - The continual decay of learning rates throughout training.
  - The need for a manually selected global learning rate.

  Adadelta is a more robust extension of Adagrad that adapts learning rates
  based on a moving window of gradient updates, instead of accumulating all
  past gradients. This way, Adadelta continues learning even when many updates
  have been done. Compared to Adagrad, in the original version of Adadelta you
  don't have to set an initial learning rate. In this version, the initial
  learning rate can be set, as in most other Keras optimizers.

  Args:
    learning_rate: Initial value for the learning rate:
      either a floating point value,
      or a `tf.keras.optimizers.schedules.LearningRateSchedule` instance.
      Defaults to 0.001.
      Note that `Adadelta` tends to benefit from higher initial learning rate
      values compared to other optimizers.
      To match the exact form in the original paper, use 1.0.
    rho: A `Tensor` or a floating point value. The decay rate.
    epsilon: Small floating point value used to maintain numerical stability.
    name: Optional name prefix for the operations created when applying
      gradients.  Defaults to `"Adadelta"`.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm and represents
      the maximum norm of each parameter;
      `"clipvalue"` (float) clips gradient by value and represents the
      maximum absolute value of each parameter.

  Reference:
    - [Zeiler, 2012](http://arxiv.org/abs/1212.5701)
  