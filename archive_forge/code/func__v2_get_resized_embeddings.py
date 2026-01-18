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
def _v2_get_resized_embeddings(self, old_embeddings: keras.layers.Embedding, new_num_tokens: int) -> keras.layers.Embedding:
    """
        Build a resized Embedding layer from a provided Embedding layer. Increasing the size will add newly initialized
        vectors at the end. Reducing the size will remove vectors from the end.

        Args:
            old_embeddings (`keras.layers.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the embedding matrix.

        Return:
            `keras.layers.Embedding`: Resized Embedding layer.
        """
    init_range = 0.02
    potential_initialization_variable_names = ['initializer_range', 'initializer_factor', 'init_std']
    for var_name in potential_initialization_variable_names:
        if hasattr(self.config, var_name):
            init_range = getattr(self.config, var_name)
    new_embeddings = keras.layers.Embedding(input_dim=new_num_tokens, output_dim=old_embeddings.output_dim, embeddings_initializer=keras.initializers.TruncatedNormal(stddev=init_range), name=old_embeddings.embeddings.name[:-13])
    new_embeddings(tf.constant([[0]]))
    if old_embeddings.input_dim >= new_num_tokens:
        init_embeddings = old_embeddings.embeddings[:new_num_tokens]
    else:
        init_embeddings = tf.concat([old_embeddings.embeddings, new_embeddings.embeddings[old_embeddings.input_dim:]], axis=0)
    new_embeddings.embeddings.assign(init_embeddings)
    return new_embeddings