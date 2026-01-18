from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import (
from .configuration_blip import BlipConfig, BlipTextConfig, BlipVisionConfig
from .modeling_tf_blip_text import BLIP_TEXT_INPUTS_DOCSTRING, TFBlipTextLMHeadModel, TFBlipTextModel
class TFBlipVisionModel(TFBlipPreTrainedModel):
    main_input_name = 'pixel_values'
    config_class = BlipVisionConfig

    def __init__(self, config: BlipVisionConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.embeddings = TFBlipVisionEmbeddings(config, name='embeddings')
        self.encoder = TFBlipEncoder(config, name='encoder')
        self.post_layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='post_layernorm')
        self.embed_dim = config.hidden_size

    def serving_output(self, output: TFBaseModelOutputWithPooling) -> TFBaseModelOutputWithPooling:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None
        return TFBaseModelOutputWithPooling(last_hidden_state=output.last_hidden_state, pooler_output=output.pooler_output, hidden_states=hs, attentions=attns)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=BlipVisionConfig)
    def call(self, pixel_values: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=None) -> Union[Tuple, TFBaseModelOutputWithPooling]:
        """
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        hidden_states = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(tf.expand_dims(pooled_output, 1))
        pooled_output = tf.squeeze(pooled_output, 1)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return TFBaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

    def get_input_embeddings(self):
        return self.embeddings

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
        if getattr(self, 'post_layernorm', None) is not None:
            with tf.name_scope(self.post_layernorm.name):
                self.post_layernorm.build([None, None, self.embed_dim])