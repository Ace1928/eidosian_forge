from __future__ import annotations
import collections.abc
import math
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_vit import ViTConfig
@keras_serializable
class TFViTMainLayer(keras.layers.Layer):
    config_class = ViTConfig

    def __init__(self, config: ViTConfig, add_pooling_layer: bool=True, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.embeddings = TFViTEmbeddings(config, name='embeddings')
        self.encoder = TFViTEncoder(config, name='encoder')
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm')
        self.pooler = TFViTPooler(config, name='pooler') if add_pooling_layer else None

    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    def call(self, pixel_values: TFModelInputType | None=None, head_mask: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, interpolate_pos_encoding: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        embedding_output = self.embeddings(pixel_values=pixel_values, interpolate_pos_encoding=interpolate_pos_encoding, training=training)
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(hidden_states=embedding_output, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(inputs=sequence_output)
        pooled_output = self.pooler(hidden_states=sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
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