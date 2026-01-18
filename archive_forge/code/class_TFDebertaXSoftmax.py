from __future__ import annotations
import math
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta import DebertaConfig
class TFDebertaXSoftmax(keras.layers.Layer):
    """
    Masked Softmax which is optimized for saving memory

    Args:
        input (`tf.Tensor`): The input tensor that will apply softmax.
        mask (`tf.Tensor`): The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax
    """

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs: tf.Tensor, mask: tf.Tensor):
        rmask = tf.logical_not(tf.cast(mask, tf.bool))
        output = tf.where(rmask, float('-inf'), inputs)
        output = stable_softmax(output, self.axis)
        output = tf.where(rmask, 0.0, output)
        return output