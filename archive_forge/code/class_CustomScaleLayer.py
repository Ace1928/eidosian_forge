import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src import backend
from keras.src import layers as keras_layers
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.layers import VersionAwareLayers
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
@keras.utils.register_keras_serializable()
class CustomScaleLayer(keras_layers.Layer):

    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def get_config(self):
        config = super().get_config()
        config.update({'scale': self.scale})
        return config

    def call(self, inputs):
        return inputs[0] + inputs[1] * self.scale