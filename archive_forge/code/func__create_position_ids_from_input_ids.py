from __future__ import annotations
import math
import random
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import TFBaseModelOutputWithPastAndCrossAttentions, TFCausalLMOutputWithCrossAttentions
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import logging
from .configuration_xglm import XGLMConfig
def _create_position_ids_from_input_ids(input_ids: tf.Tensor, past_key_values_length: int, padding_idx: Optional[int]) -> tf.Tensor:
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    """
    mask = tf.where(input_ids != padding_idx, 1, 0)
    incremental_indices = (tf.cast(tf.cumsum(mask, axis=1), dtype=mask.dtype) + past_key_values_length) * mask
    return tf.cast(incremental_indices, dtype=tf.int64) + padding_idx