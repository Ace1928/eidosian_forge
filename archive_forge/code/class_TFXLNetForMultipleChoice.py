from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_xlnet import XLNetConfig
@add_start_docstrings('\n    XLNET Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a\n    softmax) e.g. for RocStories/SWAG tasks.\n    ', XLNET_START_DOCSTRING)
class TFXLNetForMultipleChoice(TFXLNetPreTrainedModel, TFMultipleChoiceLoss):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFXLNetMainLayer(config, name='transformer')
        self.sequence_summary = TFSequenceSummary(config, initializer_range=config.initializer_range, name='sequence_summary')
        self.logits_proj = keras.layers.Dense(1, kernel_initializer=get_initializer(config.initializer_range), name='logits_proj')
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFXLNetForMultipleChoiceOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, input_mask: np.ndarray | tf.Tensor | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, mems: np.ndarray | tf.Tensor | None=None, perm_mask: np.ndarray | tf.Tensor | None=None, target_mapping: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, use_mems: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: bool=False) -> Union[TFXLNetForMultipleChoiceOutput, Tuple[tf.Tensor]]:
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
        flat_input_mask = tf.reshape(input_mask, (-1, seq_length)) if input_mask is not None else None
        flat_inputs_embeds = tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3])) if inputs_embeds is not None else None
        transformer_outputs = self.transformer(flat_input_ids, flat_attention_mask, mems, perm_mask, target_mapping, flat_token_type_ids, flat_input_mask, head_mask, flat_inputs_embeds, use_mems, output_attentions, output_hidden_states, return_dict=return_dict, training=training)
        output = transformer_outputs[0]
        logits = self.sequence_summary(output)
        logits = self.logits_proj(logits)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)
        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFXLNetForMultipleChoiceOutput(loss=loss, logits=reshaped_logits, mems=transformer_outputs.mems, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

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
        if getattr(self, 'logits_proj', None) is not None:
            with tf.name_scope(self.logits_proj.name):
                self.logits_proj.build([None, None, self.config.d_model])