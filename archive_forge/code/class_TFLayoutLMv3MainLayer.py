from __future__ import annotations
import collections
import math
from typing import List, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_layoutlmv3 import LayoutLMv3Config
@keras_serializable
class TFLayoutLMv3MainLayer(keras.layers.Layer):
    config_class = LayoutLMv3Config

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        if config.text_embed:
            self.embeddings = TFLayoutLMv3TextEmbeddings(config, name='embeddings')
        if config.visual_embed:
            self.patch_embed = TFLayoutLMv3PatchEmbeddings(config, name='patch_embed')
            self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
            self.dropout = keras.layers.Dropout(config.hidden_dropout_prob, name='dropout')
            if config.has_relative_attention_bias or config.has_spatial_attention_bias:
                image_size = config.input_size // config.patch_size
                self.init_visual_bbox(image_size=(image_size, image_size))
            self.norm = keras.layers.LayerNormalization(epsilon=1e-06, name='norm')
        self.encoder = TFLayoutLMv3Encoder(config, name='encoder')

    def build(self, input_shape=None):
        if self.config.visual_embed:
            image_size = self.config.input_size // self.config.patch_size
            self.cls_token = self.add_weight(shape=(1, 1, self.config.hidden_size), initializer='zeros', trainable=True, dtype=tf.float32, name='cls_token')
            self.pos_embed = self.add_weight(shape=(1, image_size * image_size + 1, self.config.hidden_size), initializer='zeros', trainable=True, dtype=tf.float32, name='pos_embed')
        if self.built:
            return
        self.built = True
        if getattr(self, 'encoder', None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, 'embeddings', None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        if getattr(self, 'patch_embed', None) is not None:
            with tf.name_scope(self.patch_embed.name):
                self.patch_embed.build(None)
        if getattr(self, 'LayerNorm', None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        if getattr(self, 'dropout', None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        if getattr(self, 'norm', None) is not None:
            with tf.name_scope(self.norm.name):
                self.norm.build([None, None, self.config.hidden_size])

    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.word_embeddings.weight = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    def init_visual_bbox(self, image_size: Tuple[int, int], max_len: int=1000):
        height, width = image_size
        visual_bbox_x = tf.range(0, max_len * (width + 1), max_len) // width
        visual_bbox_x = tf.expand_dims(visual_bbox_x, axis=0)
        visual_bbox_x = tf.tile(visual_bbox_x, [width, 1])
        visual_bbox_y = tf.range(0, max_len * (height + 1), max_len) // height
        visual_bbox_y = tf.expand_dims(visual_bbox_y, axis=1)
        visual_bbox_y = tf.tile(visual_bbox_y, [1, height])
        visual_bbox = tf.stack([visual_bbox_x[:, :-1], visual_bbox_y[:-1], visual_bbox_x[:, 1:], visual_bbox_y[1:]], axis=-1)
        visual_bbox = tf.reshape(visual_bbox, [-1, 4])
        cls_token_box = tf.constant([[1, 1, max_len - 1, max_len - 1]], dtype=tf.int32)
        self.visual_bbox = tf.concat([cls_token_box, visual_bbox], axis=0)

    def calculate_visual_bbox(self, batch_size: int, dtype: tf.DType):
        visual_bbox = tf.expand_dims(self.visual_bbox, axis=0)
        visual_bbox = tf.tile(visual_bbox, [batch_size, 1, 1])
        visual_bbox = tf.cast(visual_bbox, dtype=dtype)
        return visual_bbox

    def embed_image(self, pixel_values: tf.Tensor) -> tf.Tensor:
        embeddings = self.patch_embed(pixel_values)
        batch_size = tf.shape(embeddings)[0]
        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
        embeddings = tf.concat([cls_tokens, embeddings], axis=1)
        if getattr(self, 'pos_embed', None) is not None:
            embeddings += self.pos_embed
        embeddings = self.norm(embeddings)
        return embeddings

    def get_extended_attention_mask(self, attention_mask: tf.Tensor) -> tf.Tensor:
        n_dims = len(attention_mask.shape)
        if n_dims == 3:
            extended_attention_mask = tf.expand_dims(attention_mask, axis=1)
        elif n_dims == 2:
            extended_attention_mask = tf.expand_dims(attention_mask, axis=1)
            extended_attention_mask = tf.expand_dims(extended_attention_mask, axis=1)
        else:
            raise ValueError(f'Wrong shape for attention_mask (shape {attention_mask.shape}).')
        extended_attention_mask = tf.cast(extended_attention_mask, self.compute_dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * LARGE_NEGATIVE
        return extended_attention_mask

    def get_head_mask(self, head_mask: tf.Tensor | None) -> Union[tf.Tensor, List[tf.Tensor | None]]:
        if head_mask is None:
            return [None] * self.config.num_hidden_layers
        n_dims = tf.rank(head_mask)
        if n_dims == 1:
            head_mask = tf.expand_dims(head_mask, axis=0)
            head_mask = tf.expand_dims(head_mask, axis=0)
            head_mask = tf.expand_dims(head_mask, axis=-1)
            head_mask = tf.expand_dims(head_mask, axis=-1)
            head_mask = tf.tile(head_mask, [self.config.num_hidden_layers, 1, 1, 1, 1])
        elif n_dims == 2:
            head_mask = tf.expand_dims(head_mask, axis=1)
            head_mask = tf.expand_dims(head_mask, axis=-1)
            head_mask = tf.expand_dims(head_mask, axis=-1)
        elif n_dims != 5:
            raise ValueError(f'Wrong shape for head_mask (shape {head_mask.shape}).')
        assert tf.rank(head_mask) == 5, f'Got head_mask rank of {tf.rank(head_mask)}, but require 5.'
        head_mask = tf.cast(head_mask, self.compute_dtype)
        return head_mask

    @unpack_inputs
    def call(self, input_ids: tf.Tensor | None=None, bbox: tf.Tensor | None=None, attention_mask: tf.Tensor | None=None, token_type_ids: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, pixel_values: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor], Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if input_ids is not None:
            input_shape = tf.shape(input_ids)
            batch_size = input_shape[0]
            seq_length = input_shape[1]
        elif inputs_embeds is not None:
            input_shape = tf.shape(inputs_embeds)
            batch_size = input_shape[0]
            seq_length = input_shape[1]
        elif pixel_values is not None:
            batch_size = tf.shape(pixel_values)[0]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds or pixel_values')
        if input_ids is not None:
            int_dtype = input_ids.dtype
        elif bbox is not None:
            int_dtype = bbox.dtype
        elif attention_mask is not None:
            int_dtype = attention_mask.dtype
        elif token_type_ids is not None:
            int_dtype = token_type_ids.dtype
        else:
            int_dtype = tf.int32
        if input_ids is not None or inputs_embeds is not None:
            if attention_mask is None:
                attention_mask = tf.ones((batch_size, seq_length), dtype=int_dtype)
            if token_type_ids is None:
                token_type_ids = tf.zeros((batch_size, seq_length), dtype=int_dtype)
            if bbox is None:
                bbox = tf.zeros((batch_size, seq_length, 4), dtype=int_dtype)
            embedding_output = self.embeddings(input_ids=input_ids, bbox=bbox, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, training=training)
        final_bbox = None
        final_position_ids = None
        if pixel_values is not None:
            visual_embeddings = self.embed_image(pixel_values)
            visual_attention_mask = tf.ones((batch_size, tf.shape(visual_embeddings)[1]), dtype=int_dtype)
            if attention_mask is None:
                attention_mask = visual_attention_mask
            else:
                attention_mask = tf.concat([attention_mask, visual_attention_mask], axis=1)
            if self.config.has_spatial_attention_bias:
                visual_bbox = self.calculate_visual_bbox(batch_size, int_dtype)
                if bbox is None:
                    final_bbox = visual_bbox
                else:
                    final_bbox = tf.concat([bbox, visual_bbox], axis=1)
            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                visual_position_ids = tf.range(0, tf.shape(visual_embeddings)[1], dtype=int_dtype)
                visual_position_ids = tf.expand_dims(visual_position_ids, axis=0)
                visual_position_ids = tf.tile(visual_position_ids, [batch_size, 1])
                if input_ids is not None or inputs_embeds is not None:
                    position_ids = tf.expand_dims(tf.range(0, seq_length, dtype=int_dtype), axis=0)
                    position_ids = tf.tile(position_ids, [batch_size, 1])
                    final_position_ids = tf.concat([position_ids, visual_position_ids], axis=1)
                else:
                    final_position_ids = visual_position_ids
            if input_ids is None and inputs_embeds is None:
                embedding_output = visual_embeddings
            else:
                embedding_output = tf.concat([embedding_output, visual_embeddings], axis=1)
            embedding_output = self.LayerNorm(embedding_output)
            embedding_output = self.dropout(embedding_output, training=training)
        elif self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
            if self.config.has_relative_attention_bias:
                position_ids = tf.expand_dims(tf.range(0, seq_length, dtype=int_dtype), axis=0)
                position_ids = tf.tile(position_ids, [batch_size, 1])
                final_position_ids = position_ids
            if self.config.has_spatial_attention_bias:
                final_bbox = bbox
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        head_mask = self.get_head_mask(head_mask)
        encoder_outputs = self.encoder(embedding_output, bbox=final_bbox, position_ids=final_position_ids, attention_mask=extended_attention_mask, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]
        return TFBaseModelOutput(last_hidden_state=sequence_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)
        return TFBaseModelOutput(last_hidden_state=sequence_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)