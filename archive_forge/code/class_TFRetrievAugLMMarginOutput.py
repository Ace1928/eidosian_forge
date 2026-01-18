from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...configuration_utils import PretrainedConfig
from ...generation import TFLogitsProcessorList
from ...modeling_tf_utils import (
from ...utils import ModelOutput, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever
@dataclass
class TFRetrievAugLMMarginOutput(ModelOutput):
    """
    Base class for retriever augmented marginalized models outputs.

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
            each vocabulary token.
        past_key_values (`List[tf.Tensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
            (see `past_key_values` input) to speed up sequential decoding.
        doc_scores (`tf.Tensor` of shape `(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
            `question_encoder_last_hidden_state`.
        retrieved_doc_embeds (`tf.Tensor` of shape `(batch_size, config.n_docs, hidden_size)`, *optional*, returned when *output_retrieved=True*):
            Embedded documents retrieved by the retriever. Is used with `question_encoder_last_hidden_state` to compute
            the `doc_scores`.
        retrieved_doc_ids (`tf.Tensor` (int32) of shape `(batch_size, config.n_docs)`, *optional*, returned when *output_retrieved=True*):
            The indexes of the embedded documents retrieved by the retriever.
        context_input_ids (`tf.Tensor`(int32) of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
        context_attention_mask (`tf.Tensor` (int32) of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
            retriever.
        question_encoder_last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
            model.
        question_enc_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings and one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
        question_enc_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_enc_last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
        generator_enc_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings and one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
        generator_enc_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_dec_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings and one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
        generator_dec_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
    """
    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    past_key_values: List[tf.Tensor] | None = None
    doc_scores: tf.Tensor | None = None
    retrieved_doc_embeds: tf.Tensor | None = None
    retrieved_doc_ids: tf.Tensor | None = None
    context_input_ids: tf.Tensor | None = None
    context_attention_mask: tf.Tensor | None = None
    question_encoder_last_hidden_state: tf.Tensor | None = None
    question_enc_hidden_states: Tuple[tf.Tensor, ...] | None = None
    question_enc_attentions: Tuple[tf.Tensor, ...] | None = None
    generator_enc_last_hidden_state: tf.Tensor | None = None
    generator_enc_hidden_states: Tuple[tf.Tensor, ...] | None = None
    generator_enc_attentions: Tuple[tf.Tensor, ...] | None = None
    generator_dec_hidden_states: Tuple[tf.Tensor, ...] | None = None
    generator_dec_attentions: Tuple[tf.Tensor, ...] | None = None