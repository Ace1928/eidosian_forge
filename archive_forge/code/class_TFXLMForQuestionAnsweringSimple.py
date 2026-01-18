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
@add_start_docstrings('\n    XLM Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layer\n    on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', XLM_START_DOCSTRING)
class TFXLMForQuestionAnsweringSimple(TFXLMPreTrainedModel, TFQuestionAnsweringLoss):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFXLMMainLayer(config, name='transformer')
        self.qa_outputs = keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.init_std), name='qa_outputs')
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, langs: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, lengths: np.ndarray | tf.Tensor | None=None, cache: Optional[Dict[str, tf.Tensor]]=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, start_positions: np.ndarray | tf.Tensor | None=None, end_positions: np.ndarray | tf.Tensor | None=None, training: bool=False) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        """
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, langs=langs, token_type_ids=token_type_ids, position_ids=position_ids, lengths=lengths, cache=cache, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = transformer_outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {'start_position': start_positions}
            labels['end_position'] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))
        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFQuestionAnsweringModelOutput(loss=loss, start_logits=start_logits, end_logits=end_logits, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'transformer', None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        if getattr(self, 'qa_outputs', None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])