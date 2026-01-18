from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_longformer import LongformerConfig
@add_start_docstrings('Longformer Model with a `language modeling` head on top.', LONGFORMER_START_DOCSTRING)
class TFLongformerForMaskedLM(TFLongformerPreTrainedModel, TFMaskedLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = ['pooler']

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.longformer = TFLongformerMainLayer(config, add_pooling_layer=False, name='longformer')
        self.lm_head = TFLongformerLMHead(config, self.longformer.embeddings, name='lm_head')

    def get_lm_head(self):
        return self.lm_head

    def get_prefix_bias_name(self):
        warnings.warn('The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.', FutureWarning)
        return self.name + '/' + self.lm_head.name

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint='allenai/longformer-base-4096', output_type=TFLongformerMaskedLMOutput, config_class=_CONFIG_FOR_DOC, mask='<mask>', expected_output="' Paris'", expected_loss=0.44)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, global_attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFLongformerMaskedLMOutput, Tuple[tf.Tensor]]:
        """
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, global_attention_mask=global_attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output, training=training)
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFLongformerMaskedLMOutput(loss=loss, logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions, global_attentions=outputs.global_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'longformer', None) is not None:
            with tf.name_scope(self.longformer.name):
                self.longformer.build(None)
        if getattr(self, 'lm_head', None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build(None)