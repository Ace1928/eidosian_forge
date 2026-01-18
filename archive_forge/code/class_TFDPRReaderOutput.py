from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFBaseModelOutputWithPooling
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, get_initializer, keras, shape_list, unpack_inputs
from ...utils import (
from ..bert.modeling_tf_bert import TFBertMainLayer
from .configuration_dpr import DPRConfig
@dataclass
class TFDPRReaderOutput(ModelOutput):
    """
    Class for outputs of [`TFDPRReaderEncoder`].

    Args:
        start_logits (`tf.Tensor` of shape `(n_passages, sequence_length)`):
            Logits of the start index of the span for each passage.
        end_logits (`tf.Tensor` of shape `(n_passages, sequence_length)`):
            Logits of the end index of the span for each passage.
        relevance_logits (`tf.Tensor` of shape `(n_passages, )`):
            Outputs of the QA classifier of the DPRReader that corresponds to the scores of each passage to answer the
            question, compared to all the other passages.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    start_logits: tf.Tensor = None
    end_logits: tf.Tensor = None
    relevance_logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor, ...] | None = None
    attentions: Tuple[tf.Tensor, ...] | None = None