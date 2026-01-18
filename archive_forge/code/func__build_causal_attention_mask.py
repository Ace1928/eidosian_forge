from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig
def _build_causal_attention_mask(self, batch_size, seq_length, dtype=tf.float32):
    diag = tf.cast(tf.fill((seq_length,), 0.0), dtype)
    to_mask = tf.cast(tf.fill((seq_length, seq_length), -10000.0), dtype)
    to_mask = tf.linalg.band_part(to_mask, 0, -1)
    to_mask = tf.linalg.set_diag(to_mask, diagonal=diag)
    return tf.broadcast_to(input=to_mask, shape=(batch_size, 1, seq_length, seq_length))