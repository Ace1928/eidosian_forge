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
@add_start_docstrings('\n    Longformer Model with a multiple choice classification head on top (a linear layer on top of the pooled output and\n    a softmax) e.g. for RocStories/SWAG tasks.\n    ', LONGFORMER_START_DOCSTRING)
class TFLongformerForMultipleChoice(TFLongformerPreTrainedModel, TFMultipleChoiceLoss):
    _keys_to_ignore_on_load_missing = ['dropout']

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.longformer = TFLongformerMainLayer(config, name='longformer')
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = keras.layers.Dense(1, kernel_initializer=get_initializer(config.initializer_range), name='classifier')
        self.config = config

    @property
    def input_signature(self):
        return {'input_ids': tf.TensorSpec((None, None, None), tf.int32, name='input_ids'), 'attention_mask': tf.TensorSpec((None, None, None), tf.int32, name='attention_mask'), 'global_attention_mask': tf.TensorSpec((None, None, None), tf.int32, name='global_attention_mask')}

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFLongformerMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, global_attention_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFLongformerMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
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
        flat_global_attention_mask = tf.reshape(global_attention_mask, (-1, shape_list(global_attention_mask)[-1])) if global_attention_mask is not None else None
        flat_inputs_embeds = tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3])) if inputs_embeds is not None else None
        outputs = self.longformer(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids, attention_mask=flat_attention_mask, head_mask=head_mask, global_attention_mask=flat_global_attention_mask, inputs_embeds=flat_inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFLongformerMultipleChoiceModelOutput(loss=loss, logits=reshaped_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, global_attentions=outputs.global_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'longformer', None) is not None:
            with tf.name_scope(self.longformer.name):
                self.longformer.build(None)
        if getattr(self, 'classifier', None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])