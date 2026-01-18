import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_efficientformer import EfficientFormerConfig
class TFEfficientFormerConvMlp(keras.layers.Layer):

    def __init__(self, config: EfficientFormerConfig, in_features: int, hidden_features: Optional[int]=None, out_features: Optional[int]=None, drop: float=0.0, **kwargs):
        super().__init__(**kwargs)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.convolution1 = keras.layers.Conv2D(filters=hidden_features, kernel_size=1, name='convolution1', padding='valid')
        self.activation = ACT2FN[config.hidden_act]
        self.convolution2 = keras.layers.Conv2D(filters=out_features, kernel_size=1, name='convolution2', padding='valid')
        self.dropout = keras.layers.Dropout(rate=drop)
        self.batchnorm_before = keras.layers.BatchNormalization(axis=-1, epsilon=config.batch_norm_eps, momentum=0.9, name='batchnorm_before')
        self.batchnorm_after = keras.layers.BatchNormalization(axis=-1, epsilon=config.batch_norm_eps, momentum=0.9, name='batchnorm_after')
        self.hidden_features = hidden_features
        self.in_features = in_features
        self.out_features = out_features

    def call(self, hidden_state: tf.Tensor, training: bool=False) -> tf.Tensor:
        hidden_state = self.convolution1(hidden_state)
        hidden_state = self.batchnorm_before(hidden_state, training=training)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.dropout(hidden_state, training=training)
        hidden_state = self.convolution2(hidden_state)
        hidden_state = self.batchnorm_after(hidden_state, training=training)
        hidden_state = self.dropout(hidden_state, training=training)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'convolution1', None) is not None:
            with tf.name_scope(self.convolution1.name):
                self.convolution1.build([None, None, None, self.in_features])
        if getattr(self, 'convolution2', None) is not None:
            with tf.name_scope(self.convolution2.name):
                self.convolution2.build([None, None, None, self.hidden_features])
        if getattr(self, 'batchnorm_before', None) is not None:
            with tf.name_scope(self.batchnorm_before.name):
                self.batchnorm_before.build([None, None, None, self.hidden_features])
        if getattr(self, 'batchnorm_after', None) is not None:
            with tf.name_scope(self.batchnorm_after.name):
                self.batchnorm_after.build([None, None, None, self.out_features])