import tensorflow as tf
from keras.src import backend
from keras.src.backend.common import KerasVariable
from keras.src.optimizers import base_optimizer
def _clip_by_norm(self, values, axes=None):
    return tf.clip_by_norm(values, self.clipnorm, axes)