from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_mobilevit import MobileViTConfig
class TFMobileViTLayer(keras.layers.Layer):
    """
    MobileViT block: https://arxiv.org/abs/2110.02178
    """

    def __init__(self, config: MobileViTConfig, in_channels: int, out_channels: int, stride: int, hidden_size: int, num_stages: int, dilation: int=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.patch_width = config.patch_size
        self.patch_height = config.patch_size
        if stride == 2:
            self.downsampling_layer = TFMobileViTInvertedResidual(config, in_channels=in_channels, out_channels=out_channels, stride=stride if dilation == 1 else 1, dilation=dilation // 2 if dilation > 1 else 1, name='downsampling_layer')
            in_channels = out_channels
        else:
            self.downsampling_layer = None
        self.conv_kxk = TFMobileViTConvLayer(config, in_channels=in_channels, out_channels=in_channels, kernel_size=config.conv_kernel_size, name='conv_kxk')
        self.conv_1x1 = TFMobileViTConvLayer(config, in_channels=in_channels, out_channels=hidden_size, kernel_size=1, use_normalization=False, use_activation=False, name='conv_1x1')
        self.transformer = TFMobileViTTransformer(config, hidden_size=hidden_size, num_stages=num_stages, name='transformer')
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm')
        self.conv_projection = TFMobileViTConvLayer(config, in_channels=hidden_size, out_channels=in_channels, kernel_size=1, name='conv_projection')
        self.fusion = TFMobileViTConvLayer(config, in_channels=2 * in_channels, out_channels=in_channels, kernel_size=config.conv_kernel_size, name='fusion')
        self.hidden_size = hidden_size

    def unfolding(self, features: tf.Tensor) -> Tuple[tf.Tensor, Dict]:
        patch_width, patch_height = (self.patch_width, self.patch_height)
        patch_area = tf.cast(patch_width * patch_height, 'int32')
        batch_size = tf.shape(features)[0]
        orig_height = tf.shape(features)[1]
        orig_width = tf.shape(features)[2]
        channels = tf.shape(features)[3]
        new_height = tf.cast(tf.math.ceil(orig_height / patch_height) * patch_height, 'int32')
        new_width = tf.cast(tf.math.ceil(orig_width / patch_width) * patch_width, 'int32')
        interpolate = new_width != orig_width or new_height != orig_height
        if interpolate:
            features = tf.image.resize(features, size=(new_height, new_width), method='bilinear')
        num_patch_width = new_width // patch_width
        num_patch_height = new_height // patch_height
        num_patches = num_patch_height * num_patch_width
        features = tf.transpose(features, [0, 3, 1, 2])
        patches = tf.reshape(features, (batch_size * channels * num_patch_height, patch_height, num_patch_width, patch_width))
        patches = tf.transpose(patches, [0, 2, 1, 3])
        patches = tf.reshape(patches, (batch_size, channels, num_patches, patch_area))
        patches = tf.transpose(patches, [0, 3, 2, 1])
        patches = tf.reshape(patches, (batch_size * patch_area, num_patches, channels))
        info_dict = {'orig_size': (orig_height, orig_width), 'batch_size': batch_size, 'channels': channels, 'interpolate': interpolate, 'num_patches': num_patches, 'num_patches_width': num_patch_width, 'num_patches_height': num_patch_height}
        return (patches, info_dict)

    def folding(self, patches: tf.Tensor, info_dict: Dict) -> tf.Tensor:
        patch_width, patch_height = (self.patch_width, self.patch_height)
        patch_area = int(patch_width * patch_height)
        batch_size = info_dict['batch_size']
        channels = info_dict['channels']
        num_patches = info_dict['num_patches']
        num_patch_height = info_dict['num_patches_height']
        num_patch_width = info_dict['num_patches_width']
        features = tf.reshape(patches, (batch_size, patch_area, num_patches, -1))
        features = tf.transpose(features, perm=(0, 3, 2, 1))
        features = tf.reshape(features, (batch_size * channels * num_patch_height, num_patch_width, patch_height, patch_width))
        features = tf.transpose(features, perm=(0, 2, 1, 3))
        features = tf.reshape(features, (batch_size, channels, num_patch_height * patch_height, num_patch_width * patch_width))
        features = tf.transpose(features, perm=(0, 2, 3, 1))
        if info_dict['interpolate']:
            features = tf.image.resize(features, size=info_dict['orig_size'], method='bilinear')
        return features

    def call(self, features: tf.Tensor, training: bool=False) -> tf.Tensor:
        if self.downsampling_layer:
            features = self.downsampling_layer(features, training=training)
        residual = features
        features = self.conv_kxk(features, training=training)
        features = self.conv_1x1(features, training=training)
        patches, info_dict = self.unfolding(features)
        patches = self.transformer(patches, training=training)
        patches = self.layernorm(patches)
        features = self.folding(patches, info_dict)
        features = self.conv_projection(features, training=training)
        features = self.fusion(tf.concat([residual, features], axis=-1), training=training)
        return features

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'conv_kxk', None) is not None:
            with tf.name_scope(self.conv_kxk.name):
                self.conv_kxk.build(None)
        if getattr(self, 'conv_1x1', None) is not None:
            with tf.name_scope(self.conv_1x1.name):
                self.conv_1x1.build(None)
        if getattr(self, 'transformer', None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        if getattr(self, 'layernorm', None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.hidden_size])
        if getattr(self, 'conv_projection', None) is not None:
            with tf.name_scope(self.conv_projection.name):
                self.conv_projection.build(None)
        if getattr(self, 'fusion', None) is not None:
            with tf.name_scope(self.fusion.name):
                self.fusion.build(None)
        if getattr(self, 'downsampling_layer', None) is not None:
            with tf.name_scope(self.downsampling_layer.name):
                self.downsampling_layer.build(None)