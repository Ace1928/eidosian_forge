from __future__ import annotations
import random
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation, glu
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_speech_to_text import Speech2TextConfig
@keras_serializable
class TFSpeech2TextEncoder(keras.layers.Layer):
    config_class = Speech2TextConfig
    '\n    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a\n    [`TFSpeech2TextEncoderLayer`].\n\n    Args:\n        config: Speech2TextConfig\n    '

    def __init__(self, config: Speech2TextConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dropout = keras.layers.Dropout(config.dropout)
        self.layerdrop = config.encoder_layerdrop
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = tf.math.sqrt(float(embed_dim)) if config.scale_embedding else 1.0
        self.conv = TFConv1dSubsampler(config, name='conv')
        self.embed_positions = TFSpeech2TextSinusoidalPositionalEmbedding(num_positions=config.max_source_positions, embedding_dim=embed_dim, padding_idx=self.padding_idx, name='embed_positions')
        self.layers = [TFSpeech2TextEncoderLayer(config, name=f'layers.{i}') for i in range(config.encoder_layers)]
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-05, name='layer_norm')

    def _get_feat_extract_output_lengths(self, input_lengths: tf.Tensor):
        """
        Computes the output length of the convolutional layers
        """
        for _ in range(self.config.num_conv_layers):
            input_lengths = (input_lengths - 1) // 2 + 1
        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        if len(attention_mask.shape) > 2:
            attention_mask = attention_mask[:, :, -1]
        subsampled_lengths = self._get_feat_extract_output_lengths(tf.math.reduce_sum(attention_mask, -1))
        bsz = shape_list(attention_mask)[0]
        indices = tf.concat((tf.expand_dims(tf.range(bsz, dtype=attention_mask.dtype), -1), tf.expand_dims(subsampled_lengths - 1, -1)), axis=-1)
        attention_mask = tf.scatter_nd(indices=indices, updates=tf.ones(bsz), shape=[bsz, feature_vector_length])
        attention_mask = tf.cast(tf.reverse(tf.math.cumsum(tf.reverse(attention_mask, [-1]), -1), [-1]), tf.int64)
        return attention_mask

    @unpack_inputs
    def call(self, input_features=None, attention_mask=None, head_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None, training=False):
        """
        Args:
            input_features (`tf.Tensor` of shape `(batch_size, sequence_length, feature_size)`):
                Float values of fbank features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the fbank features,
                padding and conversion into a tensor of floats. See [`~Speech2TextFeatureExtractor.__call__`]
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`tf.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, `optional):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        if input_features is None:
            raise ValueError('You have to specify input_features')
        inputs_embeds = self.conv(input_features)
        inputs_embeds = self.embed_scale * inputs_embeds
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(tf.shape(inputs_embeds)[1], attention_mask)
            padding_mask = tf.cast(tf.math.not_equal(attention_mask, 1), tf.int64)
        else:
            padding_mask = tf.zeros(tf.shape(inputs_embeds)[:-1], dtype=tf.int64)
        embed_pos = self.embed_positions(padding_mask)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.dropout(hidden_states, training=training)
        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask)
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if head_mask is not None:
            tf.debugging.assert_equal(shape_list(head_mask)[0], len(self.layers), message=f'The head_mask should be specified for {len(self.layers)} layers, but it is for {shape_list(head_mask)[0]}.')
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if training and dropout_probability < self.layerdrop:
                continue
            hidden_states, attn = encoder_layer(hidden_states, attention_mask, head_mask[idx] if head_mask is not None else None, training=training)
            if output_attentions:
                all_attentions += (attn,)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'conv', None) is not None:
            with tf.name_scope(self.conv.name):
                self.conv.build(None)
        if getattr(self, 'embed_positions', None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        if getattr(self, 'layer_norm', None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])
        if getattr(self, 'layers', None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)