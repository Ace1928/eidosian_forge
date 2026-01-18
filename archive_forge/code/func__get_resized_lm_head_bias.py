from __future__ import annotations
import functools
import gc
import inspect
import json
import os
import pickle
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import h5py
import numpy as np
import tensorflow as tf
from packaging.version import parse
from . import DataCollatorWithPadding, DefaultDataCollator
from .activations_tf import get_tf_activation
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import GenerationConfig, TFGenerationMixin
from .tf_utils import (
from .utils import (
from .utils.hub import convert_file_size_to_int, get_checkpoint_shard_files
def _get_resized_lm_head_bias(self, old_lm_head_bias, new_num_tokens):
    """
        Build a resized bias from the old ones. Increasing the size will add newly initialized vectors at the end.
        Reducing the size will remove vectors from the end

        Args:
            old_lm_head_bias (`tf.Variable`):
                Old lm head bias to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns None

        Return:
            `tf.Variable`: Pointer to the resized bias.
        """
    new_lm_head_bias = {}
    for attr, weight in old_lm_head_bias.items():
        first_dim, old_num_tokens = (None, shape_list(weight)[0]) if tf.rank(weight) == 1 else shape_list(weight)
        size_diff = new_num_tokens - old_num_tokens
        final_shape = [new_num_tokens] if first_dim is None else [first_dim, new_num_tokens]
        if tf.math.greater(size_diff, 0):
            padding_shape = [[0, size_diff]] if first_dim is None else [[0, 0], [0, size_diff]]
            current_bias = tf.pad(weight.value(), tf.convert_to_tensor(padding_shape), constant_values=-1)
            num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
            mask_shape = [num_tokens_to_copy] if first_dim is None else [1, num_tokens_to_copy]
            bias_mask = tf.fill(tf.convert_to_tensor(mask_shape), True)
            bias_mask = tf.pad(bias_mask, tf.convert_to_tensor(padding_shape), constant_values=False)
        else:
            slice_from = [0] if first_dim is None else [0, 0]
            current_bias = tf.slice(weight.value(), tf.convert_to_tensor(slice_from), tf.convert_to_tensor(final_shape))
            bias_mask = tf.fill(tf.convert_to_tensor(final_shape), True)
        new_bias = self.add_weight(shape=final_shape, initializer='zeros', trainable=True, name=weight.name.split(':')[0])
        init_bias = tf.where(bias_mask, current_bias, new_bias.value())
        new_bias.assign(init_bias)
        new_lm_head_bias[attr] = new_bias
    return new_lm_head_bias