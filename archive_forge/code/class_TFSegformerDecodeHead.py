from __future__ import annotations
import math
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import TFBaseModelOutput, TFSemanticSegmenterOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_segformer import SegformerConfig
class TFSegformerDecodeHead(TFSegformerPreTrainedModel):

    def __init__(self, config: SegformerConfig, **kwargs):
        super().__init__(config, **kwargs)
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = TFSegformerMLP(config=config, input_dim=config.hidden_sizes[i], name=f'linear_c.{i}')
            mlps.append(mlp)
        self.mlps = mlps
        self.linear_fuse = keras.layers.Conv2D(filters=config.decoder_hidden_size, kernel_size=1, use_bias=False, name='linear_fuse')
        self.batch_norm = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9, name='batch_norm')
        self.activation = keras.layers.Activation('relu')
        self.dropout = keras.layers.Dropout(config.classifier_dropout_prob)
        self.classifier = keras.layers.Conv2D(filters=config.num_labels, kernel_size=1, name='classifier')
        self.config = config

    def call(self, encoder_hidden_states: tf.Tensor, training: bool=False) -> tf.Tensor:
        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.mlps):
            if self.config.reshape_last_stage is False and len(shape_list(encoder_hidden_state)) == 3:
                height = tf.math.sqrt(tf.cast(shape_list(encoder_hidden_state)[1], tf.float32))
                height = width = tf.cast(height, tf.int32)
                channel_dim = shape_list(encoder_hidden_state)[-1]
                encoder_hidden_state = tf.reshape(encoder_hidden_state, (-1, height, width, channel_dim))
            encoder_hidden_state = tf.transpose(encoder_hidden_state, perm=[0, 2, 3, 1])
            height, width = shape_list(encoder_hidden_state)[1:3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            channel_dim = shape_list(encoder_hidden_state)[-1]
            encoder_hidden_state = tf.reshape(encoder_hidden_state, (-1, height, width, channel_dim))
            temp_state = tf.transpose(encoder_hidden_states[0], perm=[0, 2, 3, 1])
            upsample_resolution = shape_list(temp_state)[1:-1]
            encoder_hidden_state = tf.image.resize(encoder_hidden_state, size=upsample_resolution, method='bilinear')
            all_hidden_states += (encoder_hidden_state,)
        hidden_states = self.linear_fuse(tf.concat(all_hidden_states[::-1], axis=-1))
        hidden_states = self.batch_norm(hidden_states, training=training)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        logits = self.classifier(hidden_states)
        return logits

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'linear_fuse', None) is not None:
            with tf.name_scope(self.linear_fuse.name):
                self.linear_fuse.build([None, None, None, self.config.decoder_hidden_size * self.config.num_encoder_blocks])
        if getattr(self, 'batch_norm', None) is not None:
            with tf.name_scope(self.batch_norm.name):
                self.batch_norm.build([None, None, None, self.config.decoder_hidden_size])
        if getattr(self, 'classifier', None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, None, self.config.decoder_hidden_size])
        if getattr(self, 'mlps', None) is not None:
            for layer in self.mlps:
                with tf.name_scope(layer.name):
                    layer.build(None)