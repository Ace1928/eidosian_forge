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
@add_start_docstrings('\n    Longformer Model with a span classification head on top for extractive question-answering tasks like SQuAD /\n    TriviaQA (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).\n    ', LONGFORMER_START_DOCSTRING)
class TFLongformerForQuestionAnswering(TFLongformerPreTrainedModel, TFQuestionAnsweringLoss):
    _keys_to_ignore_on_load_unexpected = ['pooler']

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.longformer = TFLongformerMainLayer(config, add_pooling_layer=False, name='longformer')
        self.qa_outputs = keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='qa_outputs')
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint='allenai/longformer-large-4096-finetuned-triviaqa', output_type=TFLongformerQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC, expected_output="' puppet'", expected_loss=0.96)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, global_attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, start_positions: np.ndarray | tf.Tensor | None=None, end_positions: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFLongformerQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        """
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        if input_ids is not None and (not isinstance(input_ids, tf.Tensor)):
            input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int64)
        elif input_ids is not None:
            input_ids = tf.cast(input_ids, tf.int64)
        if attention_mask is not None and (not isinstance(attention_mask, tf.Tensor)):
            attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.int64)
        elif attention_mask is not None:
            attention_mask = tf.cast(attention_mask, tf.int64)
        if global_attention_mask is not None and (not isinstance(global_attention_mask, tf.Tensor)):
            global_attention_mask = tf.convert_to_tensor(global_attention_mask, dtype=tf.int64)
        elif global_attention_mask is not None:
            global_attention_mask = tf.cast(global_attention_mask, tf.int64)
        if global_attention_mask is None and input_ids is not None:
            if shape_list(tf.where(input_ids == self.config.sep_token_id))[0] != 3 * shape_list(input_ids)[0]:
                logger.warning(f'There should be exactly three separator tokens: {self.config.sep_token_id} in every sample for questions answering. You might also consider to set `global_attention_mask` manually in the forward function to avoid this. This is most likely an error. The global attention is disabled for this forward pass.')
                global_attention_mask = tf.cast(tf.fill(shape_list(input_ids), value=0), tf.int64)
            else:
                logger.warning_once('Initializing global attention on question tokens...')
                sep_token_indices = tf.where(input_ids == self.config.sep_token_id)
                sep_token_indices = tf.cast(sep_token_indices, dtype=tf.int64)
                global_attention_mask = _compute_global_attention_mask(shape_list(input_ids), sep_token_indices)
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, global_attention_mask=global_attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
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
            output = (start_logits, end_logits) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFLongformerQuestionAnsweringModelOutput(loss=loss, start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, global_attentions=outputs.global_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'longformer', None) is not None:
            with tf.name_scope(self.longformer.name):
                self.longformer.build(None)
        if getattr(self, 'qa_outputs', None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])