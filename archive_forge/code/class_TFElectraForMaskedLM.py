from __future__ import annotations
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_electra import ElectraConfig
@add_start_docstrings('\n    Electra model with a language modeling head on top.\n\n    Even though both the discriminator and generator may be loaded into this model, the generator is the only model of\n    the two to have been trained for the masked language modeling task.\n    ', ELECTRA_START_DOCSTRING)
class TFElectraForMaskedLM(TFElectraPreTrainedModel, TFMaskedLanguageModelingLoss):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.electra = TFElectraMainLayer(config, name='electra')
        self.generator_predictions = TFElectraGeneratorPredictions(config, name='generator_predictions')
        if isinstance(config.hidden_act, str):
            self.activation = get_tf_activation(config.hidden_act)
        else:
            self.activation = config.hidden_act
        self.generator_lm_head = TFElectraMaskedLMHead(config, self.electra.embeddings, name='generator_lm_head')

    def get_lm_head(self):
        return self.generator_lm_head

    def get_prefix_bias_name(self):
        warnings.warn('The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.', FutureWarning)
        return self.name + '/' + self.generator_lm_head.name

    @unpack_inputs
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint='google/electra-small-generator', output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC, mask='[MASK]', expected_output="'paris'", expected_loss=1.22)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        """
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        generator_hidden_states = self.electra(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        generator_sequence_output = generator_hidden_states[0]
        prediction_scores = self.generator_predictions(generator_sequence_output, training=training)
        prediction_scores = self.generator_lm_head(prediction_scores, training=training)
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)
        if not return_dict:
            output = (prediction_scores,) + generator_hidden_states[1:]
            return (loss,) + output if loss is not None else output
        return TFMaskedLMOutput(loss=loss, logits=prediction_scores, hidden_states=generator_hidden_states.hidden_states, attentions=generator_hidden_states.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'electra', None) is not None:
            with tf.name_scope(self.electra.name):
                self.electra.build(None)
        if getattr(self, 'generator_predictions', None) is not None:
            with tf.name_scope(self.generator_predictions.name):
                self.generator_predictions.build(None)
        if getattr(self, 'generator_lm_head', None) is not None:
            with tf.name_scope(self.generator_lm_head.name):
                self.generator_lm_head.build(None)