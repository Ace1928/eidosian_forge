from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_resnet import ResNetConfig
class TFResNetEncoder(keras.layers.Layer):

    def __init__(self, config: ResNetConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.stages = [TFResNetStage(config, config.embedding_size, config.hidden_sizes[0], stride=2 if config.downsample_in_first_stage else 1, depth=config.depths[0], name='stages.0')]
        for i, (in_channels, out_channels, depth) in enumerate(zip(config.hidden_sizes, config.hidden_sizes[1:], config.depths[1:])):
            self.stages.append(TFResNetStage(config, in_channels, out_channels, depth=depth, name=f'stages.{i + 1}'))

    def call(self, hidden_state: tf.Tensor, output_hidden_states: bool=False, return_dict: bool=True, training: bool=False) -> TFBaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None
        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)
            hidden_state = stage_module(hidden_state, training=training)
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)
        if not return_dict:
            return tuple((v for v in [hidden_state, hidden_states] if v is not None))
        return TFBaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'stages', None) is not None:
            for layer in self.stages:
                with tf.name_scope(layer.name):
                    layer.build(None)