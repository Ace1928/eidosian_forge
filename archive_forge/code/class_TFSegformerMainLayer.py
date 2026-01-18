from __future__ import annotations
import math
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import TFBaseModelOutput, TFSemanticSegmenterOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_segformer import SegformerConfig
@keras_serializable
class TFSegformerMainLayer(keras.layers.Layer):
    config_class = SegformerConfig

    def __init__(self, config: SegformerConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.encoder = TFSegformerEncoder(config, name='encoder')

    @unpack_inputs
    def call(self, pixel_values: tf.Tensor, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[Tuple, TFBaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        encoder_outputs = self.encoder(pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = encoder_outputs[0]
        sequence_output = tf.transpose(sequence_output, perm=[0, 3, 1, 2])
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])
        if not return_dict:
            if tf.greater(len(encoder_outputs[1:]), 0):
                transposed_encoder_outputs = tuple((tf.transpose(v, perm=[0, 3, 1, 2]) for v in encoder_outputs[1:][0]))
                return (sequence_output,) + (transposed_encoder_outputs,)
            else:
                return (sequence_output,) + encoder_outputs[1:]
        return TFBaseModelOutput(last_hidden_state=sequence_output, hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'encoder', None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)