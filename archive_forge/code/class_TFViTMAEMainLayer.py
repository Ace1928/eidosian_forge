from __future__ import annotations
import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import TFBaseModelOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_vit_mae import ViTMAEConfig
@keras_serializable
class TFViTMAEMainLayer(keras.layers.Layer):
    config_class = ViTMAEConfig

    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.embeddings = TFViTMAEEmbeddings(config, name='embeddings')
        self.encoder = TFViTMAEEncoder(config, name='encoder')
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm')

    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    def call(self, pixel_values: TFModelInputType | None=None, noise: tf.Tensor=None, head_mask: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFViTMAEModelOutput, Tuple[tf.Tensor]]:
        embedding_output, mask, ids_restore = self.embeddings(pixel_values=pixel_values, training=training, noise=noise)
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(embedding_output, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(inputs=sequence_output)
        if not return_dict:
            return (sequence_output, mask, ids_restore) + encoder_outputs[1:]
        return TFViTMAEModelOutput(last_hidden_state=sequence_output, mask=mask, ids_restore=ids_restore, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

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