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