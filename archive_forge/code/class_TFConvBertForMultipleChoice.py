from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_convbert import ConvBertConfig
@add_start_docstrings('\n    ConvBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a\n    softmax) e.g. for RocStories/SWAG tasks.\n    ', CONVBERT_START_DOCSTRING)
class TFConvBertForMultipleChoice(TFConvBertPreTrainedModel, TFMultipleChoiceLoss):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.convbert = TFConvBertMainLayer(config, name='convbert')
        self.sequence_summary = TFSequenceSummary(config, initializer_range=config.initializer_range, name='sequence_summary')
        self.classifier = keras.layers.Dense(1, kernel_initializer=get_initializer(config.initializer_range), name='classifier')
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: tf.Tensor | None=None, training: Optional[bool]=False) -> Union[Tuple, TFMultipleChoiceModelOutput]:
        """
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
        flat_inputs_embeds = tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3])) if inputs_embeds is not None else None
        outputs = self.convbert(flat_input_ids, flat_attention_mask, flat_token_type_ids, flat_position_ids, head_mask, flat_inputs_embeds, output_attentions, output_hidden_states, return_dict=return_dict, training=training)
        logits = self.sequence_summary(outputs[0], training=training)
        logits = self.classifier(logits)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFMultipleChoiceModelOutput(loss=loss, logits=reshaped_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'convbert', None) is not None:
            with tf.name_scope(self.convbert.name):
                self.convbert.build(None)
        if getattr(self, 'sequence_summary', None) is not None:
            with tf.name_scope(self.sequence_summary.name):
                self.sequence_summary.build(None)
        if getattr(self, 'classifier', None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])