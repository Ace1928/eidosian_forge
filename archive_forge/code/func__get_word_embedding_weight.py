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
def _get_word_embedding_weight(model, embedding_layer):
    if isinstance(embedding_layer, tf.Tensor):
        return embedding_layer
    embeds = getattr(embedding_layer, 'weight', None)
    if embeds is not None:
        return embeds
    embeds = getattr(embedding_layer, 'decoder', None)
    if embeds is not None:
        return embeds
    model.build_in_name_scope()
    embeds = getattr(embedding_layer, 'weight', None)
    if embeds is not None:
        return embeds
    embeds = getattr(embedding_layer, 'decoder', None)
    if embeds is not None:
        return embeds
    return None