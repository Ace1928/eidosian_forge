import abc
import math
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.saving import serialization_lib
from keras.src.saving.legacy import serialization as legacy_serialization
from tensorflow.python.util.tf_export import keras_export
def _warmup_function(self, step, warmup_steps, warmup_target, initial_learning_rate):
    with tf.name_scope(self.name or 'CosineDecay'):
        completed_fraction = step / warmup_steps
        total_step_delta = warmup_target - initial_learning_rate
        return total_step_delta * completed_fraction + initial_learning_rate