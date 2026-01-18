from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_deit import DeiTConfig
@keras_serializable
class TFDeiTMainLayer(keras.layers.Layer):
    config_class = DeiTConfig

    def __init__(self, config: DeiTConfig, add_pooling_layer: bool=True, use_mask_token: bool=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.embeddings = TFDeiTEmbeddings(config, use_mask_token=use_mask_token, name='embeddings')
        self.encoder = TFDeiTEncoder(config, name='encoder')
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm')
        self.pooler = TFDeiTPooler(config, name='pooler') if add_pooling_layer else None

    def get_input_embeddings(self) -> TFDeiTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    def get_head_mask(self, head_mask):
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers
        return head_mask

    @unpack_inputs
    def call(self, pixel_values: tf.Tensor | None=None, bool_masked_pos: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor, ...]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        pixel_values = tf.transpose(pixel_values, (0, 2, 3, 1))
        head_mask = self.get_head_mask(head_mask)
        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos, training=training)
        encoder_outputs = self.encoder(embedding_output, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output, training=training)
        pooled_output = self.pooler(sequence_output, training=training) if self.pooler is not None else None
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]
        return TFBaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'embeddings', None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        if getattr(self, 'encoder', None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, 'layernorm', None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.config.hidden_size])
        if getattr(self, 'pooler', None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)