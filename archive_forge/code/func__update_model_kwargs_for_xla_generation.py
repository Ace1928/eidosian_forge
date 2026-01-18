import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice
from ..modeling_tf_outputs import TFCausalLMOutputWithPast, TFSeq2SeqLMOutput
from ..models.auto import (
from ..tf_utils import shape_list, stable_softmax
from ..utils import ModelOutput, logging
from .configuration_utils import GenerationConfig
from .tf_logits_process import (
def _update_model_kwargs_for_xla_generation(self, model_outputs: ModelOutput, model_kwargs: Dict[str, Any], cur_len: int, max_length: int, batch_size: int, is_encoder_decoder: bool=False, batch_axis: int=0):

    def _initialize_attention(model_kwargs, num_padding_values, is_encoder_decoder):
        """initializes the appropriate attention mask -- encoder-decoder models use `decoder_attention_mask`"""
        if is_encoder_decoder:
            decoder_attention_mask = tf.concat([tf.ones((batch_size, 1), dtype=tf.int32), tf.zeros((batch_size, num_padding_values), dtype=tf.int32), tf.ones((batch_size, 1), dtype=tf.int32)], axis=1)
            mask = {'decoder_attention_mask': decoder_attention_mask}
        else:
            attention_mask = model_kwargs.pop('attention_mask')
            attention_mask = tf.concat([attention_mask, tf.zeros((batch_size, num_padding_values), dtype=attention_mask.dtype), tf.ones((batch_size, 1), dtype=attention_mask.dtype)], axis=1)
            mask = {'attention_mask': attention_mask}
        return mask

    def _update_attention(model_kwargs, new_past_index, is_encoder_decoder):
        """updates the appropriate attention mask -- encoder-decoder models use `decoder_attention_mask`"""
        update_start = tf.constant([0, 1], dtype=tf.int32) * new_past_index
        if is_encoder_decoder:
            decoder_attention_mask = model_kwargs.pop('decoder_attention_mask')
            decoder_attention_mask_update_slice = tf.ones((batch_size, 1), dtype=decoder_attention_mask.dtype)
            decoder_attention_mask = dynamic_update_slice(decoder_attention_mask, decoder_attention_mask_update_slice, update_start)
            mask = {'decoder_attention_mask': decoder_attention_mask}
        else:
            attention_mask = model_kwargs.pop('attention_mask')
            attention_mask_update_slice = tf.ones((batch_size, 1), dtype=attention_mask.dtype)
            attention_mask = dynamic_update_slice(attention_mask, attention_mask_update_slice, update_start)
            mask = {'attention_mask': attention_mask}
        return mask

    def _initialize_past(past_key_values, num_padding_values, batch_axis):
        """initialize past_key_values with zeros -- the structure depends on `batch_axis`"""
        if batch_axis == 0:
            padding_values = tf.constant([[0, 0], [0, 0], [0, num_padding_values], [0, 0]], dtype=tf.int32)
            new_past = ()
            for past_layer in past_key_values:
                new_past_layer = list(past_layer)
                for i in range(len(new_past_layer[:2])):
                    new_past_layer[i] = tf.pad(past_layer[i], padding_values)
                new_past += (tuple(new_past_layer),)
        else:
            padding_values = tf.scatter_nd(indices=[[3, 1]], updates=[num_padding_values], shape=(5, 2))
            new_past = list(past_key_values)
            for i in range(len(past_key_values)):
                new_past[i] = tf.pad(past_key_values[i], padding_values)
        return new_past

    def _update_past(past_key_values, new_past_index, batch_axis):
        if batch_axis == 0:
            slice_start_base = tf.constant([0, 0, 1, 0])
            new_past = ()
            for past_layer in past_key_values:
                new_past_layer = list(past_layer)
                for i in range(len(new_past_layer[:2])):
                    update_slice = past_layer[i][:, :, -1:]
                    new_past_layer[i] = dynamic_update_slice(past_layer[i][:, :, :-1], update_slice, slice_start_base * new_past_index)
                new_past += (tuple(new_past_layer),)
        else:
            slice_start_base = tf.constant([0, 0, 0, 1, 0])
            new_past = [None for _ in range(len(past_key_values))]
            for i in range(len(past_key_values)):
                update_slice = past_key_values[i][:, :, :, -1:]
                new_past[i] = dynamic_update_slice(past_key_values[i][:, :, :, :-1], update_slice, slice_start_base * new_past_index)
        return new_past
    past_key_values = self._extract_past_from_model_output(model_outputs)
    if past_key_values is None:
        raise ValueError(f'No known `past_key_values variable` found in model outputs (model outputs keys: {list(model_outputs.keys())})')
    is_past_initialized = model_kwargs.pop('past_key_values', None) is not None
    if not is_past_initialized:
        num_padding_values = max_length - cur_len - 1
        mask = _initialize_attention(model_kwargs, num_padding_values, is_encoder_decoder)
        new_past = _initialize_past(past_key_values, num_padding_values, batch_axis)
    else:
        new_past_index = cur_len - 2
        mask = _update_attention(model_kwargs, new_past_index, is_encoder_decoder)
        new_past = _update_past(past_key_values, new_past_index, batch_axis)
    model_kwargs.update(mask)
    model_kwargs['past_key_values'] = tuple(new_past)
    return model_kwargs