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
def _create_position_ids_from_inputs_embeds(inputs_embeds: tf.Tensor, past_key_values_length: int, padding_idx: Optional[int]) -> tf.Tensor:
    """
    Args:
    We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
        inputs_embeds: tf.Tensor
    Returns: tf.Tensor
    """
    input_shape = shape_list(inputs_embeds)[:-1]
    sequence_length = input_shape[1]
    position_ids = tf.range(padding_idx + 1, sequence_length + padding_idx + 1, dtype=tf.int64)
    return tf.broadcast_to(tf.expand_dims(position_ids, axis=0), input_shape) + past_key_values_length