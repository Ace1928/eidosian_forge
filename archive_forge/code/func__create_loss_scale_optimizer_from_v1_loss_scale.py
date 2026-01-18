import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import optimizers
from keras.src.dtensor import utils as dtensor_utils
from keras.src.optimizers import optimizer
from keras.src.optimizers import utils as optimizer_utils
from keras.src.optimizers.legacy import optimizer_v2
from keras.src.saving import serialization_lib
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.tf_export import keras_export
def _create_loss_scale_optimizer_from_v1_loss_scale(optimizer, loss_scale):
    """Creates an LSO from a tf.compat.v1.mixed_precision.LossScale.

    This is only used to pass to
    `tf.__internal__.mixed_precision.register_loss_scale_wrapper` below, which
    is called so that
    `tf.compat.v1.mixed_precision.enable_mixed_precision_graph_rewrite` can
    wrap a Keras optimizer with a LossScaleOptimizer.

    Args:
      optimizer: An OptimizerV2 instance.
      loss_scale: A `tf.compat.v1.mixed_precision.LossScale` instance

    Returns:
      A LossScaleOptimizer that wraps `optimizer` and uses the same loss scaling
      algorithm as `loss_scale`.
    """
    if isinstance(loss_scale, (int, float)):
        return LossScaleOptimizer(optimizer, dynamic=False, initial_scale=loss_scale)
    elif isinstance(loss_scale, tf.compat.v1.mixed_precision.FixedLossScale):
        ls_val = loss_scale._loss_scale_value
        return LossScaleOptimizer(optimizer, dynamic=False, initial_scale=ls_val)
    elif loss_scale == 'dynamic':
        return LossScaleOptimizer(optimizer)
    elif isinstance(loss_scale, tf.compat.v1.mixed_precision.DynamicLossScale):
        if loss_scale.multiplier != 2:
            raise ValueError(f'When passing a DynamicLossScale to "loss_scale", DynamicLossScale.multiplier must be 2. Got: {loss_scale}')
        return LossScaleOptimizer(optimizer, initial_scale=loss_scale.initial_loss_scale, dynamic_growth_steps=loss_scale.increment_period)
    elif isinstance(loss_scale, tf.compat.v1.mixed_precision.LossScale):
        raise TypeError(f'Passing a LossScale that is not a FixedLossScale or a DynamicLossScale is not supported. Got: {loss_scale}')
    else:
        raise ValueError(f'Invalid value passed to loss_scale. loss_scale must be the string "dynamic" (recommended), an int, a float, a FixedLossScale, or a DynamicLossScale. Got value: {loss_scale}')