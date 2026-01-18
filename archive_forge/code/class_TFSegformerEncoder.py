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
class TFSegformerEncoder(keras.layers.Layer):

    def __init__(self, config: SegformerConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        drop_path_decays = [x.numpy() for x in tf.linspace(0.0, config.drop_path_rate, sum(config.depths))]
        embeddings = []
        for i in range(config.num_encoder_blocks):
            embeddings.append(TFSegformerOverlapPatchEmbeddings(patch_size=config.patch_sizes[i], stride=config.strides[i], num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1], hidden_size=config.hidden_sizes[i], name=f'patch_embeddings.{i}'))
        self.embeddings = embeddings
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(TFSegformerLayer(config, hidden_size=config.hidden_sizes[i], num_attention_heads=config.num_attention_heads[i], drop_path=drop_path_decays[cur + j], sequence_reduction_ratio=config.sr_ratios[i], mlp_ratio=config.mlp_ratios[i], name=f'block.{i}.{j}'))
            blocks.append(layers)
        self.block = blocks
        self.layer_norms = [keras.layers.LayerNormalization(epsilon=1e-05, name=f'layer_norm.{i}') for i in range(config.num_encoder_blocks)]

    def call(self, pixel_values: tf.Tensor, output_attentions: Optional[bool]=False, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True, training: bool=False) -> Union[Tuple, TFBaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        batch_size = shape_list(pixel_values)[0]
        hidden_states = pixel_values
        for idx, x in enumerate(zip(self.embeddings, self.block, self.layer_norms)):
            embedding_layer, block_layer, norm_layer = x
            hidden_states, height, width = embedding_layer(hidden_states)
            for i, blk in enumerate(block_layer):
                layer_outputs = blk(hidden_states, height, width, output_attentions, training=training)
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
            hidden_states = norm_layer(hidden_states)
            if idx != len(self.embeddings) - 1 or (idx == len(self.embeddings) - 1 and self.config.reshape_last_stage):
                num_channels = shape_list(hidden_states)[-1]
                hidden_states = tf.reshape(hidden_states, (batch_size, height, width, num_channels))
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layer_norms', None) is not None:
            for layer, shape in zip(self.layer_norms, self.config.hidden_sizes):
                with tf.name_scope(layer.name):
                    layer.build([None, None, shape])
        if getattr(self, 'block', None) is not None:
            for block in self.block:
                for layer in block:
                    with tf.name_scope(layer.name):
                        layer.build(None)
        if getattr(self, 'embeddings', None) is not None:
            for layer in self.embeddings:
                with tf.name_scope(layer.name):
                    layer.build(None)