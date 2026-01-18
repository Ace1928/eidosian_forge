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
class TFEfficientFormerMeta3D(keras.layers.Layer):

    def __init__(self, config: EfficientFormerConfig, dim: int, drop_path: float=0.0, **kwargs):
        super().__init__(**kwargs)
        self.token_mixer = TFEfficientFormerSelfAttention(dim=config.dim, key_dim=config.key_dim, num_heads=config.num_attention_heads, attention_ratio=config.attention_ratio, resolution=config.resolution, name='token_mixer', config=config)
        self.dim = dim
        self.config = config
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm1')
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm2')
        mlp_hidden_dim = int(dim * config.mlp_expansion_ratio)
        self.mlp = TFEfficientFormerDenseMlp(config, in_features=dim, hidden_features=mlp_hidden_dim, name='mlp')
        self.drop_path = TFEfficientFormerDropPath(drop_path) if drop_path > 0.0 else keras.layers.Activation('linear', name='drop_path')
        self.config = config

    def build(self, input_shape=None):
        self.layer_scale_1 = None
        self.layer_scale_2 = None
        if self.config.use_layer_scale:
            self.layer_scale_1 = self.add_weight(shape=(self.dim,), initializer=keras.initializers.Constant(value=self.config.layer_scale_init_value), trainable=True, name='layer_scale_1')
            self.layer_scale_2 = self.add_weight(shape=(self.dim,), initializer=keras.initializers.Constant(value=self.config.layer_scale_init_value), trainable=True, name='layer_scale_2')
        if self.built:
            return
        self.built = True
        if getattr(self, 'token_mixer', None) is not None:
            with tf.name_scope(self.token_mixer.name):
                self.token_mixer.build(None)
        if getattr(self, 'layernorm1', None) is not None:
            with tf.name_scope(self.layernorm1.name):
                self.layernorm1.build([None, None, self.dim])
        if getattr(self, 'layernorm2', None) is not None:
            with tf.name_scope(self.layernorm2.name):
                self.layernorm2.build([None, None, self.dim])
        if getattr(self, 'mlp', None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        if getattr(self, 'drop_path', None) is not None:
            with tf.name_scope(self.drop_path.name):
                self.drop_path.build(None)

    def call(self, hidden_states: tf.Tensor, output_attentions: bool=False, training: bool=False) -> Tuple[tf.Tensor]:
        self_attention_outputs = self.token_mixer(hidden_states=self.layernorm1(hidden_states, training=training), output_attentions=output_attentions, training=training)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        if self.config.use_layer_scale:
            layer_output = hidden_states + self.drop_path(tf.expand_dims(tf.expand_dims(self.layer_scale_1, 0), 0) * attention_output, training=training)
            layer_output = layer_output + self.drop_path(tf.expand_dims(tf.expand_dims(self.layer_scale_2, 0), 0) * self.mlp(hidden_states=self.layernorm2(inputs=layer_output, training=training), training=training), training=training)
        else:
            layer_output = hidden_states + self.drop_path(attention_output, training=training)
            layer_output = layer_output + self.drop_path(self.mlp(hidden_states=self.layernorm2(inputs=layer_output, training=training), training=training), training=training)
        outputs = (layer_output,) + outputs
        return outputs