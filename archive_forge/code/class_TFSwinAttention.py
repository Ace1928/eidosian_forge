from __future__ import annotations
import collections.abc
import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import (
from .configuration_swin import SwinConfig
class TFSwinAttention(keras.layers.Layer):

    def __init__(self, config: SwinConfig, dim: int, num_heads: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.self = TFSwinSelfAttention(config, dim, num_heads, name='self')
        self.self_output = TFSwinSelfOutput(config, dim, name='output')
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        Prunes heads of the model. See base class PreTrainedModel heads: dict of {layer_num: list of heads to prune in
        this layer}
        """
        raise NotImplementedError

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, output_attentions: bool=False, training: bool=False) -> tf.Tensor:
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions, training=training)
        attention_output = self.self_output(self_outputs[0], hidden_states, training=training)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'self', None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        if getattr(self, 'self_output', None) is not None:
            with tf.name_scope(self.self_output.name):
                self.self_output.build(None)