from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_mobilebert import MobileBertConfig
class TFMobileBertPreTrainingLoss:
    """
    Loss function suitable for BERT-like pretraining, that is, the task of pretraining a language model by combining
    NSP + MLM. .. note:: Any label of -100 will be ignored (along with the corresponding logits) in the loss
    computation.
    """

    def hf_compute_loss(self, labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        unmasked_lm_losses = loss_fn(y_true=tf.nn.relu(labels['labels']), y_pred=logits[0])
        lm_loss_mask = tf.cast(labels['labels'] != -100, dtype=unmasked_lm_losses.dtype)
        masked_lm_losses = unmasked_lm_losses * lm_loss_mask
        reduced_masked_lm_loss = tf.reduce_sum(masked_lm_losses) / tf.reduce_sum(lm_loss_mask)
        unmasked_ns_loss = loss_fn(y_true=tf.nn.relu(labels['next_sentence_label']), y_pred=logits[1])
        ns_loss_mask = tf.cast(labels['next_sentence_label'] != -100, dtype=unmasked_ns_loss.dtype)
        masked_ns_loss = unmasked_ns_loss * ns_loss_mask
        reduced_masked_ns_loss = tf.reduce_sum(masked_ns_loss) / tf.reduce_sum(ns_loss_mask)
        return tf.reshape(reduced_masked_lm_loss + reduced_masked_ns_loss, (1,))