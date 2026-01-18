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
def _v2_get_resized_lm_head_bias(self, old_lm_head_bias: Dict[str, tf.Variable], new_num_tokens: int) -> Dict[str, tf.Tensor]:
    """
        Build a resized bias from the old ones. Increasing the size will add newly initialized vectors at the end.
        Reducing the size will remove vectors from the end

        Args:
            old_lm_head_bias (`Dict[str, tf.Variable]`):
                Old lm head bias to be resized.
            new_num_tokens (`int`):
                New number of tokens in the linear matrix. Increasing the size will add newly initialized vectors at
                the end. Reducing the size will remove vectors from the end.

        Return:
            `tf.Tensor`: Values for the resized bias.
        """
    new_lm_head_bias = {}
    for attr, weight in old_lm_head_bias.items():
        first_dim, old_num_tokens = (None, shape_list(weight)[0]) if tf.rank(weight) == 1 else shape_list(weight)
        size_diff = new_num_tokens - old_num_tokens
        if old_num_tokens > new_num_tokens:
            new_bias = weight.value()[..., :new_num_tokens]
        else:
            padding_shape = [[0, size_diff]] if first_dim is None else [[0, 0], [0, size_diff]]
            new_bias = tf.pad(weight.value(), tf.convert_to_tensor(padding_shape))
        new_lm_head_bias[attr] = new_bias
    return new_lm_head_bias