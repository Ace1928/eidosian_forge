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
class TFEfficientFormerIntermediateStage(keras.layers.Layer):

    def __init__(self, config: EfficientFormerConfig, index: int, **kwargs):
        super().__init__(**kwargs)
        self.meta4D_layers = TFEfficientFormerMeta4DLayers(config=config, stage_idx=index, name='meta4D_layers')

    def call(self, hidden_states: tf.Tensor, training: bool=False) -> Tuple[tf.Tensor]:
        hidden_states = self.meta4D_layers(hidden_states=hidden_states, training=training)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'meta4D_layers', None) is not None:
            with tf.name_scope(self.meta4D_layers.name):
                self.meta4D_layers.build(None)