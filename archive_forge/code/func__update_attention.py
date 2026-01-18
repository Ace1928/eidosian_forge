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