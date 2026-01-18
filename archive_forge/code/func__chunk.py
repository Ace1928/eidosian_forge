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
def _chunk(hidden_states, window_overlap):
    """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
    batch_size, seq_length, hidden_dim = shape_list(hidden_states)
    num_output_chunks = 2 * (seq_length // (2 * window_overlap)) - 1
    frame_hop_size = window_overlap * hidden_dim
    frame_size = 2 * frame_hop_size
    hidden_states = tf.reshape(hidden_states, (batch_size, seq_length * hidden_dim))
    chunked_hidden_states = tf.signal.frame(hidden_states, frame_size, frame_hop_size)
    tf.debugging.assert_equal(shape_list(chunked_hidden_states), [batch_size, num_output_chunks, frame_size], message=f'Make sure chunking is correctly applied. `Chunked hidden states should have output  dimension {[batch_size, frame_size, num_output_chunks]}, but got {shape_list(chunked_hidden_states)}.')
    chunked_hidden_states = tf.reshape(chunked_hidden_states, (batch_size, num_output_chunks, 2 * window_overlap, hidden_dim))
    return chunked_hidden_states