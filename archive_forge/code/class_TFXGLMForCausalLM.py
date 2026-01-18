from __future__ import annotations
import math
import random
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import TFBaseModelOutputWithPastAndCrossAttentions, TFCausalLMOutputWithCrossAttentions
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import logging
from .configuration_xglm import XGLMConfig
@add_start_docstrings('\n    The XGLM Model transformer with a language modeling head on top (linear layer with weights tied to the input\n    embeddings).\n    ', XGLM_START_DOCSTRING)
class TFXGLMForCausalLM(TFXGLMPreTrainedModel, TFCausalLanguageModelingLoss):
    base_model_prefix = 'model'
    _keys_to_ignore_on_load_missing = ['model.embed_positions.weights', 'lm_head.weight']
    _keys_to_ignore_on_save = ['model.embed_positions.weights']

    def __init__(self, config: XGLMConfig, embed_tokens: Optional[TFSharedEmbeddings]=None, *inputs: Any, **kwargs: Any) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.model = TFXGLMMainLayer(config, embed_tokens=embed_tokens, name='model')
        self.lm_head = keras.layers.Dense(config.vocab_size, use_bias=False, kernel_initializer=get_initializer(config.init_std), name='lm_head')
        self.config = config

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, inputs, past_key_values=None, use_cache=None, **kwargs):
        if past_key_values:
            inputs = tf.expand_dims(inputs[:, -1], -1)
        position_ids = kwargs.get('position_ids', None)
        attention_mask = kwargs.get('attention_mask', None)
        if attention_mask is not None and position_ids is None:
            position_ids = tf.math.cumsum(attention_mask, axis=-1, exclusive=True)
            if past_key_values:
                position_ids = tf.expand_dims(position_ids[:, -1], -1)
        return {'input_ids': inputs, 'attention_mask': attention_mask, 'position_ids': position_ids, 'past_key_values': past_key_values, 'use_cache': use_cache}

    @unpack_inputs
    @add_start_docstrings_to_model_forward(XGLM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFCausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFCausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, encoder_hidden_states: np.ndarray | tf.Tensor | None=None, encoder_attention_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, cross_attn_head_mask: np.ndarray | tf.Tensor | None=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, labels: np.ndarray | tf.Tensor | None=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False, **kwargs: Any) -> Union[TFCausalLMOutputWithCrossAttentions, Tuple[tf.Tensor]]:
        """
        labels (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, head_mask=head_mask, cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            labels = tf.concat([labels[:, 1:], tf.fill((labels.shape[0], 1), tf.cast(self.config.pad_token_id, labels.dtype))], axis=-1)
            loss = self.hf_compute_loss(labels, lm_logits)
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return TFCausalLMOutputWithCrossAttentions(loss=loss, logits=lm_logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'model', None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
        if getattr(self, 'lm_head', None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build([None, None, self.config.hidden_size])

    def tf_to_pt_weight_rename(self, tf_weight):
        if tf_weight == 'lm_head.weight':
            return (tf_weight, 'model.embed_tokens.weight')
        else:
            return (tf_weight,)