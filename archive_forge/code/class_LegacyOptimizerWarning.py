from keras.src.api_export import keras_export
from keras.src.optimizers.adadelta import Adadelta
from keras.src.optimizers.adafactor import Adafactor
from keras.src.optimizers.adagrad import Adagrad
from keras.src.optimizers.adam import Adam
from keras.src.optimizers.adamax import Adamax
from keras.src.optimizers.adamw import AdamW
from keras.src.optimizers.ftrl import Ftrl
from keras.src.optimizers.lion import Lion
from keras.src.optimizers.loss_scale_optimizer import LossScaleOptimizer
from keras.src.optimizers.nadam import Nadam
from keras.src.optimizers.optimizer import Optimizer
from keras.src.optimizers.rmsprop import RMSprop
from keras.src.optimizers.sgd import SGD
from keras.src.saving import serialization_lib
@keras_export(['keras.optimizers.legacy.Adagrad', 'keras.optimizers.legacy.Adam', 'keras.optimizers.legacy.Ftrl', 'keras.optimizers.legacy.RMSprop', 'keras.optimizers.legacy.SGD', 'keras.optimizers.legacy.Optimizer'])
class LegacyOptimizerWarning:

    def __init__(self, *args, **kwargs):
        raise ImportError('`keras.optimizers.legacy` is not supported in Keras 3. When using `tf.keras`, to continue using a `tf.keras.optimizers.legacy` optimizer, you can install the `tf_keras` package (Keras 2) and set the environment variable `TF_USE_LEGACY_KERAS=True` to configure TensorFlow to use `tf_keras` when accessing `tf.keras`.')