from __future__ import annotations
import os
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import logging
from .configuration_esm import EsmConfig
def _compute_cos_sin(self, x, seq_dimension=2):
    seq_len = tf.shape(x)[seq_dimension]
    t = tf.range(seq_len, dtype=self.inv_freq.dtype)
    freqs = tf.einsum('i, j -> ij', t, self.inv_freq)
    emb = tf.concat((freqs, freqs), axis=-1)[None, None, :, :]
    return (tf.cos(emb), tf.sin(emb))