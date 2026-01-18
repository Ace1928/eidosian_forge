from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import (
from .configuration_lxmert import LxmertConfig
@dataclass
class TFLxmertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`LxmertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `tf.Tensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cross_relationship_score (`tf.Tensor` of shape `(batch_size, 2)`):
            Prediction scores of the textual matching objective (classification) head (scores of True/False
            continuation before SoftMax).
        question_answering_score (`tf.Tensor` of shape `(batch_size, n_qa_answers)`):
            Prediction scores of question answering objective (classification).
        language_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for input features + one for the output of each cross-modality layer) of shape
            `(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for input features + one for the output of each cross-modality layer) of shape
            `(batch_size, sequence_length, hidden_size)`.
        language_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        vision_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_encoder_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.

    """
    loss: tf.Tensor | None = None
    prediction_logits: tf.Tensor | None = None
    cross_relationship_score: tf.Tensor | None = None
    question_answering_score: tf.Tensor | None = None
    language_hidden_states: Tuple[tf.Tensor] | None = None
    vision_hidden_states: Tuple[tf.Tensor] | None = None
    language_attentions: Tuple[tf.Tensor] | None = None
    vision_attentions: Tuple[tf.Tensor] | None = None
    cross_encoder_attentions: Tuple[tf.Tensor] | None = None