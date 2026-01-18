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
def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
    """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
    if head_mask.shape.rank == 1:
        head_mask = head_mask[None, None, :, None, None]
        head_mask = tf.repeat(head_mask, repeats=num_hidden_layers, axis=0)
    elif head_mask.shape.rank == 2:
        head_mask = head_mask[:, None, :, None, None]
    assert head_mask.shape.rank == 5, f'head_mask.dim != 5, instead {head_mask.dim()}'
    head_mask = tf.cast(head_mask, tf.float32)
    return head_mask