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
class LossScaleOptimizerMetaclass(type):
    """Metaclass that delegates LossScaleOptimizer instance creation.

    This metaclass causes a LossScaleOptimizer or LossScaleOptimizerV3 to be
    created when a BaseLossScaleOptimizer is constructed. As a result, when a
    user creates a loss scale optimizer with
    `tf.keras.mixed_precision.LossScaleOptimizer(opt)`, either a
    LossScaleOptimizer or LossScaleOptimizerV3 will be created, depending on the
    type of `opt`.
    """

    def __call__(cls, inner_optimizer, *args, **kwargs):
        if cls is not BaseLossScaleOptimizer:
            return super(LossScaleOptimizerMetaclass, cls).__call__(inner_optimizer, *args, **kwargs)
        if isinstance(inner_optimizer, optimizer_v2.OptimizerV2):
            return LossScaleOptimizer(inner_optimizer, *args, **kwargs)
        elif isinstance(inner_optimizer, optimizer.Optimizer):
            return LossScaleOptimizerV3(inner_optimizer, *args, **kwargs)
        msg = f'"inner_optimizer" must be an instance of `tf.keras.optimizers.Optimizer` or `tf.keras.optimizers.experimental.Optimizer`, but got: {inner_optimizer}.'
        raise TypeError(msg)