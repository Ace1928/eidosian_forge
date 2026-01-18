from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import one_device_strategy
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.mixed_precision import loss_scale as keras_loss_scale_module
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.optimizer_v2 import utils as optimizer_utils
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import base_delegate
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.training.experimental import mixed_precision
from tensorflow.python.util import nest
class LossScaleOptimizerV1(LossScaleOptimizer):
    """An deprecated optimizer that applies loss scaling.

  Warning: This class is deprecated and will be removed in a future version of
  TensorFlow. Please use the non-experimental class
  `tf.keras.mixed_precision.LossScaleOptimizer` instead.

  This class is identical to the non-experimental
  `keras.mixed_precision.LossScaleOptimizer` except its constructor takes
  different arguments. For this class (the experimental version), the
  constructor takes a `loss_scale` argument.  For the non-experimental class,
  the constructor encodes the loss scaling information in multiple arguments.
  Note that unlike this class, the non-experimental class does not accept a
  `tf.compat.v1.mixed_precision.LossScale`, which is deprecated.

  If you currently use this class, you should switch to the non-experimental
  `tf.keras.mixed_precision.LossScaleOptimizer` instead. We show several
  examples of converting the use of the experimental class to the equivalent
  non-experimental class.

  >>> # In all of the examples below, `opt1` and `opt2` are identical
  >>> opt1 = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
  ...     tf.keras.optimizers.SGD(), loss_scale='dynamic')
  >>> opt2 = tf.keras.mixed_precision.LossScaleOptimizer(
  ...     tf.keras.optimizers.SGD())
  >>> assert opt1.get_config() == opt2.get_config()

  >>> opt1 = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
  ...     tf.keras.optimizers.SGD(), loss_scale=123)
  >>> # dynamic=False indicates to use fixed loss scaling. initial_scale=123
  >>> # refers to the initial loss scale, which is the single fixed loss scale
  >>> # when dynamic=False.
  >>> opt2 = tf.keras.mixed_precision.LossScaleOptimizer(
  ...     tf.keras.optimizers.SGD(), dynamic=False, initial_scale=123)
  >>> assert opt1.get_config() == opt2.get_config()

  >>> loss_scale = tf.compat.v1.mixed_precision.experimental.DynamicLossScale(
  ...     initial_loss_scale=2048, increment_period=500)
  >>> opt1 = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
  ...     tf.keras.optimizers.SGD(), loss_scale=loss_scale)
  >>> opt2 = tf.keras.mixed_precision.LossScaleOptimizer(
  ...     tf.keras.optimizers.SGD(), initial_scale=2048,
  ...     dynamic_growth_steps=500)
  >>> assert opt1.get_config() == opt2.get_config()

  Make sure to also switch from this class to the non-experimental class in
  isinstance checks, if you have any. If you do not do this, your model may run
  into hard-to-debug issues, as the experimental `LossScaleOptimizer` subclasses
  the non-experimental `LossScaleOptimizer`, but not vice versa. It is safe to
  switch isinstance checks to the non-experimental `LossScaleOptimizer` even
  before using the non-experimental `LossScaleOptimizer`.

  >>> opt1 = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
  ...     tf.keras.optimizers.SGD(), loss_scale='dynamic')
  >>> # The experimental class subclasses the non-experimental class
  >>> isinstance(opt1, tf.keras.mixed_precision.LossScaleOptimizer)
  True
  >>> opt2 = tf.keras.mixed_precision.LossScaleOptimizer(
  ...     tf.keras.optimizers.SGD())
  >>> # The non-experimental class does NOT subclass the experimental class.
  >>> isinstance(opt2, tf.keras.mixed_precision.experimental.LossScaleOptimizer)
  False

  Args:
    optimizer: The Optimizer instance to wrap.
    loss_scale: The loss scale to scale the loss and gradients. This can
      either be an int/float to use a fixed loss scale, the string "dynamic"
      to use dynamic loss scaling, or an instance of a LossScale. The string
      "dynamic" equivalent to passing `DynamicLossScale()`, and passing an
      int/float is equivalent to passing a FixedLossScale with the given loss
      scale. If a DynamicLossScale is passed, DynamicLossScale.multiplier must
      be 2 (the default).
  """

    def __init__(self, optimizer, loss_scale):
        warn_msg_prefix = 'tf.keras.mixed_precision.experimental.LossScaleOptimizer is deprecated. Please use tf.keras.mixed_precision.LossScaleOptimizer instead. '
        if isinstance(loss_scale, dict):
            loss_scale = keras_loss_scale_module.deserialize(loss_scale)
        if isinstance(loss_scale, (int, float)):
            tf_logging.warning(warn_msg_prefix + 'For example:\n  opt = tf.keras.mixed_precision.LossScaleOptimizer(opt, dynamic=False, initial_scale={})'.format(loss_scale))
            super(LossScaleOptimizerV1, self).__init__(optimizer, dynamic=False, initial_scale=loss_scale)
        elif isinstance(loss_scale, loss_scale_module.FixedLossScale):
            ls_val = loss_scale._loss_scale_value
            tf_logging.warning(warn_msg_prefix + 'For example:\n  opt = tf.keras.mixed_precision.LossScaleOptimizer(opt, dynamic=False, initial_scale={})'.format(ls_val))
            super(LossScaleOptimizerV1, self).__init__(optimizer, dynamic=False, initial_scale=ls_val)
        elif loss_scale == 'dynamic':
            tf_logging.warning(warn_msg_prefix + 'For example:\n  opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)')
            super(LossScaleOptimizerV1, self).__init__(optimizer)
        elif isinstance(loss_scale, loss_scale_module.DynamicLossScale):
            kwargs = {}
            extra_arguments = ''
            if loss_scale.initial_loss_scale != _DEFAULT_INITIAL_SCALE:
                kwargs['initial_scale'] = loss_scale.initial_loss_scale
                extra_arguments += ', initial_scale=%s' % loss_scale.initial_loss_scale
            if loss_scale.increment_period != _DEFAULT_GROWTH_STEPS:
                kwargs['dynamic_growth_steps'] = loss_scale.increment_period
                extra_arguments += ', dynamic_growth_steps=%s' % loss_scale.increment_period
            if loss_scale.multiplier != 2:
                raise ValueError('When passing a DynamicLossScale to "loss_scale", DynamicLossScale.multiplier must be 2. Got: %s' % (loss_scale,))
            tf_logging.warning(warn_msg_prefix + 'Note that the non-experimental LossScaleOptimizer does not take a DynamicLossScale but instead takes the dynamic configuration directly in the constructor. For example:\n  opt = tf.keras.mixed_precision.LossScaleOptimizer(opt{})\n'.format(extra_arguments))
            super(LossScaleOptimizerV1, self).__init__(optimizer, **kwargs)
        elif isinstance(loss_scale, loss_scale_module.LossScale):
            raise TypeError('Passing a LossScale that is not a FixedLossScale or a DynamicLossScale is no longer supported. Got: {}'.format(loss_scale))
        else:
            raise ValueError('Invalid value passed to loss_scale. loss_scale must be the string "dynamic" (recommended), an int, a float, a FixedLossScale, or a DynamicLossScale. Got value: {}'.format(loss_scale))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()
        if 'loss_scale' in config:
            config['loss_scale'] = keras_loss_scale_module.deserialize(config['loss_scale'])
            if isinstance(config['loss_scale'], loss_scale_module.DynamicLossScale) and config['loss_scale'].multiplier != 2:
                raise ValueError('Cannot deserialize LossScaleOptimizer with a DynamicLossScale whose multiplier is not 2. Got DynamicLossScale: %s' % (config['loss_scale'],))
            config['optimizer'] = optimizers.deserialize(config['optimizer'], custom_objects=custom_objects)
            return cls(**config)
        if config['dynamic']:
            config['loss_scale'] = loss_scale_module.DynamicLossScale(config['initial_scale'], config['dynamic_growth_steps'], multiplier=2)
        else:
            config['loss_scale'] = loss_scale_module.FixedLossScale(config['initial_scale'])
        del config['dynamic']
        del config['initial_scale']
        del config['dynamic_growth_steps']
        config['optimizer'] = optimizers.deserialize(config.pop('inner_optimizer'), custom_objects=custom_objects)
        return cls(**config)