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
@add_start_docstrings('\n    XLM Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a\n    softmax) e.g. for RocStories/SWAG tasks.\n    ', XLM_START_DOCSTRING)
class TFXLMForMultipleChoice(TFXLMPreTrainedModel, TFMultipleChoiceLoss):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFXLMMainLayer(config, name='transformer')
        self.sequence_summary = TFSequenceSummary(config, initializer_range=config.init_std, name='sequence_summary')
        self.logits_proj = keras.layers.Dense(1, kernel_initializer=get_initializer(config.initializer_range), name='logits_proj')
        self.config = config

    @property
    def dummy_inputs(self):
        """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        if self.config.use_lang_emb and self.config.n_langs > 1:
            return {'input_ids': tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS, dtype=tf.int32), 'langs': tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS, dtype=tf.int32)}
        else:
            return {'input_ids': tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS, dtype=tf.int32)}

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, langs: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, lengths: np.ndarray | tf.Tensor | None=None, cache: Optional[Dict[str, tf.Tensor]]=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, training: bool=False) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
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
        flat_langs = tf.reshape(langs, (-1, seq_length)) if langs is not None else None
        flat_inputs_embeds = tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3])) if inputs_embeds is not None else None
        if lengths is not None:
            logger.warning('The `lengths` parameter cannot be used with the XLM multiple choice models. Please use the attention mask instead.')
            lengths = None
        transformer_outputs = self.transformer(flat_input_ids, flat_attention_mask, flat_langs, flat_token_type_ids, flat_position_ids, lengths, cache, head_mask, flat_inputs_embeds, output_attentions, output_hidden_states, return_dict=return_dict, training=training)
        output = transformer_outputs[0]
        logits = self.sequence_summary(output)
        logits = self.logits_proj(logits)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)
        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFMultipleChoiceModelOutput(loss=loss, logits=reshaped_logits, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

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
                self.logits_proj.build([None, None, self.config.num_labels])