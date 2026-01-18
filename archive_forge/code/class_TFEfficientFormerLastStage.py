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
class TFEfficientFormerLastStage(keras.layers.Layer):

    def __init__(self, config: EfficientFormerConfig, **kwargs):
        super().__init__(**kwargs)
        self.meta4D_layers = TFEfficientFormerMeta4DLayers(config=config, stage_idx=-1, name='meta4D_layers')
        self.flat = TFEfficientFormerFlat(name='flat')
        self.meta3D_layers = TFEfficientFormerMeta3DLayers(config, name='meta3D_layers')

    def call(self, hidden_states: tf.Tensor, output_attentions: bool=False, training: bool=False) -> Tuple[tf.Tensor]:
        hidden_states = self.meta4D_layers(hidden_states=hidden_states, training=training)
        hidden_states = self.flat(hidden_states=hidden_states)
        hidden_states = self.meta3D_layers(hidden_states=hidden_states, output_attentions=output_attentions, training=training)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'meta4D_layers', None) is not None:
            with tf.name_scope(self.meta4D_layers.name):
                self.meta4D_layers.build(None)
        if getattr(self, 'flat', None) is not None:
            with tf.name_scope(self.flat.name):
                self.flat.build(None)
        if getattr(self, 'meta3D_layers', None) is not None:
            with tf.name_scope(self.meta3D_layers.name):
                self.meta3D_layers.build(None)