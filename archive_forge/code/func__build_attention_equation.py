import collections
import math
import string
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.engine.base_layer import Layer
from keras.src.layers import activation
from keras.src.layers import core
from keras.src.layers import regularization
from keras.src.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
def _build_attention_equation(rank, attn_axes):
    """Builds einsum equations for the attention computation.

    Query, key, value inputs after projection are expected to have the shape as:
    `(bs, <non-attention dims>, <attention dims>, num_heads, channels)`.
    `bs` and `<non-attention dims>` are treated as `<batch dims>`.

    The attention operations can be generalized:
    (1) Query-key dot product:
    `(<batch dims>, <query attention dims>, num_heads, channels), (<batch dims>,
    <key attention dims>, num_heads, channels) -> (<batch dims>,
    num_heads, <query attention dims>, <key attention dims>)`
    (2) Combination:
    `(<batch dims>, num_heads, <query attention dims>, <key attention dims>),
    (<batch dims>, <value attention dims>, num_heads, channels) -> (<batch
    dims>, <query attention dims>, num_heads, channels)`

    Args:
        rank: Rank of query, key, value tensors.
        attn_axes: List/tuple of axes, `[-1, rank)`,
            that attention will be applied to.

    Returns:
        Einsum equations.
    """
    target_notation = _CHR_IDX[:rank]
    batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
    letter_offset = rank
    source_notation = ''
    for i in range(rank):
        if i in batch_dims or i == rank - 1:
            source_notation += target_notation[i]
        else:
            source_notation += _CHR_IDX[letter_offset]
            letter_offset += 1
    product_notation = ''.join([target_notation[i] for i in batch_dims] + [target_notation[i] for i in attn_axes] + [source_notation[i] for i in attn_axes])
    dot_product_equation = '%s,%s->%s' % (source_notation, target_notation, product_notation)
    attn_scores_rank = len(product_notation)
    combine_equation = '%s,%s->%s' % (product_notation, source_notation, target_notation)
    return (dot_product_equation, combine_equation, attn_scores_rank)