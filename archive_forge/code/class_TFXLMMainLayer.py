from __future__ import annotations
import itertools
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_xlm import XLMConfig
@keras_serializable
class TFXLMMainLayer(keras.layers.Layer):
    config_class = XLMConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.return_dict = config.use_return_dict
        self.is_encoder = config.is_encoder
        self.is_decoder = not config.is_encoder
        if self.is_decoder:
            raise NotImplementedError('Currently XLM can only be used as an encoder')
        self.causal = config.causal
        self.n_langs = config.n_langs
        self.use_lang_emb = config.use_lang_emb
        self.n_words = config.n_words
        self.eos_index = config.eos_index
        self.pad_index = config.pad_index
        self.dim = config.emb_dim
        self.hidden_dim = self.dim * 4
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.max_position_embeddings = config.max_position_embeddings
        self.embed_init_std = config.embed_init_std
        if self.dim % self.n_heads != 0:
            raise ValueError('transformer dim must be a multiple of n_heads')
        self.dropout = keras.layers.Dropout(config.dropout)
        self.attention_dropout = keras.layers.Dropout(config.attention_dropout)
        if config.sinusoidal_embeddings:
            raise NotImplementedError
        self.embeddings = TFSharedEmbeddings(self.n_words, self.dim, initializer_range=config.embed_init_std, name='embeddings')
        self.layer_norm_emb = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm_emb')
        self.attentions = []
        self.layer_norm1 = []
        self.ffns = []
        self.layer_norm2 = []
        for i in range(self.n_layers):
            self.attentions.append(TFXLMMultiHeadAttention(self.n_heads, self.dim, config=config, name=f'attentions_._{i}'))
            self.layer_norm1.append(keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name=f'layer_norm1_._{i}'))
            self.ffns.append(TFXLMTransformerFFN(self.dim, self.hidden_dim, self.dim, config=config, name=f'ffns_._{i}'))
            self.layer_norm2.append(keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name=f'layer_norm2_._{i}'))
        if hasattr(config, 'pruned_heads'):
            pruned_heads = config.pruned_heads.copy().items()
            config.pruned_heads = {}
            for layer, heads in pruned_heads:
                if self.attentions[int(layer)].n_heads == config.n_heads:
                    self.prune_heads({int(layer): list(map(int, heads))})

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        with tf.name_scope('position_embeddings'):
            self.position_embeddings = self.add_weight(name='embeddings', shape=[self.max_position_embeddings, self.dim], initializer=get_initializer(self.embed_init_std))
        if self.n_langs > 1 and self.use_lang_emb:
            with tf.name_scope('lang_embeddings'):
                self.lang_embeddings = self.add_weight(name='embeddings', shape=[self.n_langs, self.dim], initializer=get_initializer(self.embed_init_std))
        if getattr(self, 'embeddings', None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        if getattr(self, 'layer_norm_emb', None) is not None:
            with tf.name_scope(self.layer_norm_emb.name):
                self.layer_norm_emb.build([None, None, self.dim])
        for layer in self.attentions:
            with tf.name_scope(layer.name):
                layer.build(None)
        for layer in self.layer_norm1:
            with tf.name_scope(layer.name):
                layer.build([None, None, self.dim])
        for layer in self.ffns:
            with tf.name_scope(layer.name):
                layer.build(None)
        for layer in self.layer_norm2:
            with tf.name_scope(layer.name):
                layer.build([None, None, self.dim])

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    def call(self, input_ids=None, attention_mask=None, langs=None, token_type_ids=None, position_ids=None, lengths=None, cache=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None, training=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            bs, slen = shape_list(input_ids)
        elif inputs_embeds is not None:
            bs, slen = shape_list(inputs_embeds)[:2]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if lengths is None:
            if input_ids is not None:
                lengths = tf.reduce_sum(tf.cast(tf.not_equal(input_ids, self.pad_index), dtype=input_ids.dtype), axis=1)
            else:
                lengths = tf.convert_to_tensor([slen] * bs)
        (tf.debugging.assert_equal(shape_list(lengths)[0], bs), f'Expected batch size {shape_list(lengths)[0]} and received batch size {bs} mismatched')
        mask, attn_mask = get_masks(slen, lengths, self.causal, padding_mask=attention_mask)
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(slen), axis=0)
            position_ids = tf.tile(position_ids, (bs, 1))
        (tf.debugging.assert_equal(shape_list(position_ids), [bs, slen]), f'Position id shape {shape_list(position_ids)} and input shape {[bs, slen]} mismatched')
        if langs is not None:
            (tf.debugging.assert_equal(shape_list(langs), [bs, slen]), f'Lang shape {shape_list(langs)} and input shape {[bs, slen]} mismatched')
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.n_layers
        if cache is not None and input_ids is not None:
            _slen = slen - cache['slen']
            input_ids = input_ids[:, -_slen:]
            position_ids = position_ids[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]
        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.embeddings.vocab_size)
            inputs_embeds = self.embeddings(input_ids)
        tensor = inputs_embeds + tf.gather(self.position_embeddings, position_ids)
        if langs is not None and self.use_lang_emb and (self.n_langs > 1):
            tensor = tensor + tf.gather(self.lang_embeddings, langs)
        if token_type_ids is not None:
            tensor = tensor + self.embeddings(token_type_ids)
        tensor = self.layer_norm_emb(tensor)
        tensor = self.dropout(tensor, training=training)
        mask = tf.cast(mask, dtype=tensor.dtype)
        tensor = tensor * tf.expand_dims(mask, axis=-1)
        hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        for i in range(self.n_layers):
            if output_hidden_states:
                hidden_states = hidden_states + (tensor,)
            attn_outputs = self.attentions[i](tensor, attn_mask, None, cache, head_mask[i], output_attentions, training=training)
            attn = attn_outputs[0]
            if output_attentions:
                attentions = attentions + (attn_outputs[1],)
            attn = self.dropout(attn, training=training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)
            tensor = tensor * tf.expand_dims(mask, axis=-1)
        if output_hidden_states:
            hidden_states = hidden_states + (tensor,)
        if cache is not None:
            cache['slen'] += tensor.size(1)
        if not return_dict:
            return tuple((v for v in [tensor, hidden_states, attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=tensor, hidden_states=hidden_states, attentions=attentions)