from __future__ import annotations
import random
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation, glu
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_speech_to_text import Speech2TextConfig
@add_start_docstrings('The bare Speech2Text Model outputting raw hidden-states without any specific head on top.', SPEECH_TO_TEXT_START_DOCSTRING)
class TFSpeech2TextModel(TFSpeech2TextPreTrainedModel):

    def __init__(self, config: Speech2TextConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = TFSpeech2TextMainLayer(config, name='model')

    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder

    @unpack_inputs
    @add_start_docstrings_to_model_forward(SPEECH_TO_TEXT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_features: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, decoder_input_ids: np.ndarray | tf.Tensor | None=None, decoder_attention_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, decoder_head_mask: np.ndarray | tf.Tensor | None=None, cross_attn_head_mask: np.ndarray | tf.Tensor | None=None, encoder_outputs: np.ndarray | tf.Tensor | None=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, decoder_inputs_embeds: np.ndarray | tf.Tensor | None=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False, **kwargs) -> Union[Tuple, TFSeq2SeqModelOutput]:
        outputs = self.model(input_features=input_features, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, head_mask=head_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, encoder_outputs=encoder_outputs, past_key_values=past_key_values, decoder_inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

    def serving_output(self, output):
        pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None
        return TFSeq2SeqModelOutput(last_hidden_state=output.last_hidden_state, past_key_values=pkv, decoder_hidden_states=dec_hs, decoder_attentions=dec_attns, cross_attentions=cross_attns, encoder_last_hidden_state=output.encoder_last_hidden_state, encoder_hidden_states=enc_hs, encoder_attentions=enc_attns)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'model', None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)