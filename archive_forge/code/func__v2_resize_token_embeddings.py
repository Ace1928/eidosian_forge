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
def _v2_resize_token_embeddings(self, new_num_tokens):
    old_embeddings = self.get_input_embeddings()
    new_embeddings = self._v2_get_resized_embeddings(old_embeddings, new_num_tokens)
    self.set_input_embeddings(new_embeddings)
    if self.get_bias() is not None:
        old_lm_head_bias = self.get_bias()
        new_lm_head_bias = self._v2_get_resized_lm_head_bias(old_lm_head_bias, new_num_tokens)
        self.set_bias(new_lm_head_bias)
    tied_weights = self.get_input_embeddings() == self.get_output_embeddings()
    if self.get_output_embeddings() is not None and (not tied_weights):
        old_lm_head_decoder = self._get_word_embedding_weight(self.get_output_embeddings())
        new_lm_head_decoder = self._get_resized_lm_head_decoder(old_lm_head_decoder, new_num_tokens)
        self.set_output_embeddings(new_lm_head_decoder)
    return self.get_input_embeddings()