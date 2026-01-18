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
class TFEfficientFormerMeta4DLayers(keras.layers.Layer):

    def __init__(self, config: EfficientFormerConfig, stage_idx: int, **kwargs):
        super().__init__(**kwargs)
        num_layers = config.depths[stage_idx] if stage_idx != -1 else config.depths[stage_idx] - config.num_meta3d_blocks
        drop_paths = [config.drop_path_rate * (block_idx + sum(config.depths[:stage_idx])) for block_idx in range(num_layers)]
        self.blocks = [TFEfficientFormerMeta4D(config=config, dim=config.hidden_sizes[stage_idx], drop_path=drop_paths[i], name=f'blocks.{i}') for i in range(len(drop_paths))]

    def call(self, hidden_states: tf.Tensor, training: bool=False) -> Tuple[tf.Tensor]:
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states=hidden_states, training=training)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'blocks', None) is not None:
            for layer in self.blocks:
                with tf.name_scope(layer.name):
                    layer.build(None)