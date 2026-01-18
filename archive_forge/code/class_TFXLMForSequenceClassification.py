from __future__ import annotations
import itertools
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_xlm import XLMConfig
@add_start_docstrings('\n    XLM Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.\n    for GLUE tasks.\n    ', XLM_START_DOCSTRING)
class TFXLMForSequenceClassification(TFXLMPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.transformer = TFXLMMainLayer(config, name='transformer')
        self.sequence_summary = TFSequenceSummary(config, initializer_range=config.init_std, name='sequence_summary')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, langs: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, lengths: np.ndarray | tf.Tensor | None=None, cache: Optional[Dict[str, tf.Tensor]]=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: bool=False) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        """
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, langs=langs, token_type_ids=token_type_ids, position_ids=position_ids, lengths=lengths, cache=cache, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        output = transformer_outputs[0]
        logits = self.sequence_summary(output)
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'transformer', None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        if getattr(self, 'sequence_summary', None) is not None:
            with tf.name_scope(self.sequence_summary.name):
                self.sequence_summary.build(None)