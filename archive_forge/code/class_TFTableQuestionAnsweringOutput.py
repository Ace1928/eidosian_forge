from __future__ import annotations
import enum
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_tapas import TapasConfig
@dataclass
class TFTableQuestionAnsweringOutput(ModelOutput):
    """
    Output type of [`TFTapasForQuestionAnswering`].

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` (and possibly `answer`, `aggregation_labels`, `numeric_values` and `numeric_values_scale` are provided)):
            Total loss as the sum of the hierarchical cell selection log-likelihood loss and (optionally) the
            semi-supervised regression loss and (optionally) supervised loss for aggregations.
        logits (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Prediction scores of the cell selection head, for every token.
        logits_aggregation (`tf.Tensor`, *optional*, of shape `(batch_size, num_aggregation_labels)`):
            Prediction scores of the aggregation head, for every aggregation operator.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """
    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    logits_aggregation: tf.Tensor | None = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None