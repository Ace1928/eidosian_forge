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
class TFEfficientFormerMeta3DLayers(keras.layers.Layer):

    def __init__(self, config: EfficientFormerConfig, **kwargs):
        super().__init__(**kwargs)
        drop_paths = [config.drop_path_rate * (block_idx + sum(config.depths[:-1])) for block_idx in range(config.num_meta3d_blocks)]
        self.blocks = [TFEfficientFormerMeta3D(config, config.hidden_sizes[-1], drop_path=drop_path, name=f'blocks.{i}') for i, drop_path in enumerate(drop_paths)]

    def call(self, hidden_states: tf.Tensor, output_attentions: bool=False, training: bool=False) -> Tuple[tf.Tensor]:
        all_attention_outputs = () if output_attentions else None
        for i, layer_module in enumerate(self.blocks):
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            hidden_states = layer_module(hidden_states=hidden_states, output_attentions=output_attentions, training=training)
            if output_attentions:
                all_attention_outputs = all_attention_outputs + (hidden_states[1],)
        if output_attentions:
            outputs = (hidden_states[0],) + all_attention_outputs
            return outputs
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'blocks', None) is not None:
            for layer in self.blocks:
                with tf.name_scope(layer.name):
                    layer.build(None)