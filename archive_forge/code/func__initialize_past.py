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