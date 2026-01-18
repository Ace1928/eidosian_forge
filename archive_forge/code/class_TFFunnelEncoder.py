from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_funnel import FunnelConfig
class TFFunnelEncoder(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.separate_cls = config.separate_cls
        self.pool_q_only = config.pool_q_only
        self.block_repeats = config.block_repeats
        self.attention_structure = TFFunnelAttentionStructure(config)
        self.blocks = [[TFFunnelLayer(config, block_index, name=f'blocks_._{block_index}_._{i}') for i in range(block_size)] for block_index, block_size in enumerate(config.block_sizes)]

    def call(self, inputs_embeds, attention_mask=None, token_type_ids=None, output_attentions=False, output_hidden_states=False, return_dict=True, training=False):
        attention_inputs = self.attention_structure.init_attention_inputs(inputs_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids, training=training)
        hidden = inputs_embeds
        all_hidden_states = (inputs_embeds,) if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for block_index, block in enumerate(self.blocks):
            pooling_flag = shape_list(hidden)[1] > (2 if self.separate_cls else 1)
            pooling_flag = pooling_flag and block_index > 0
            pooled_hidden = tf.zeros(shape_list(hidden))
            if pooling_flag:
                pooled_hidden, attention_inputs = self.attention_structure.pre_attention_pooling(hidden, attention_inputs)
            for layer_index, layer in enumerate(block):
                for repeat_index in range(self.block_repeats[block_index]):
                    do_pooling = repeat_index == 0 and layer_index == 0 and pooling_flag
                    if do_pooling:
                        query = pooled_hidden
                        key = value = hidden if self.pool_q_only else pooled_hidden
                    else:
                        query = key = value = hidden
                    layer_output = layer(query, key, value, attention_inputs, output_attentions=output_attentions, training=training)
                    hidden = layer_output[0]
                    if do_pooling:
                        attention_inputs = self.attention_structure.post_attention_pooling(attention_inputs)
                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)
        if not return_dict:
            return tuple((v for v in [hidden, all_hidden_states, all_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        for block in self.blocks:
            for layer in block:
                with tf.name_scope(layer.name):
                    layer.build(None)