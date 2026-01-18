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
class TFEfficientFormerPatchEmbeddings(keras.layers.Layer):
    """
    This class performs downsampling between two stages. For the input tensor with the shape [batch_size, num_channels,
    height, width] it produces output tensor with the shape [batch_size, num_channels, height/stride, width/stride]
    """

    def __init__(self, config: EfficientFormerConfig, num_channels: int, embed_dim: int, apply_norm: bool=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.padding = keras.layers.ZeroPadding2D(padding=config.downsample_pad)
        self.projection = keras.layers.Conv2D(filters=embed_dim, kernel_size=config.downsample_patch_size, strides=config.downsample_stride, padding='valid', name='projection')
        self.norm = keras.layers.BatchNormalization(axis=-1, epsilon=config.batch_norm_eps, momentum=0.9, name='norm') if apply_norm else tf.identity
        self.embed_dim = embed_dim

    def call(self, pixel_values: tf.Tensor, training: bool=False) -> tf.Tensor:
        tf.debugging.assert_shapes([(pixel_values, (..., None, None, self.num_channels))], message='Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        embeddings = self.projection(self.padding(pixel_values))
        embeddings = self.norm(embeddings, training=training)
        return embeddings

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'projection', None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, None, self.num_channels])
        if getattr(self, 'norm', None) is not None:
            if hasattr(self.norm, 'name'):
                with tf.name_scope(self.norm.name):
                    self.norm.build([None, None, None, self.embed_dim])