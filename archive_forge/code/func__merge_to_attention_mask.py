from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_longformer import LongformerConfig
@staticmethod
def _merge_to_attention_mask(attention_mask: tf.Tensor, global_attention_mask: tf.Tensor):
    if attention_mask is not None:
        attention_mask = attention_mask * (global_attention_mask + 1)
    else:
        attention_mask = global_attention_mask + 1
    return attention_mask