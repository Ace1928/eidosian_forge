import inspect
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from ..tf_utils import stable_softmax
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
def _create_score_penalties(self, input_ids: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    logit_penalties = tf.gather(logits, input_ids, axis=1, batch_dims=1)
    logit_penalties = tf.where(logit_penalties > 0, 1 / self.penalty, logit_penalties)
    logit_penalties = tf.where(logit_penalties < 0, self.penalty, logit_penalties)
    token_penalties = tf.ones(logits.shape)
    batch_size = input_ids.shape[0]
    seq_len = tf.shape(input_ids)[1]
    indexable_prev_input_ids = tf.concat((tf.expand_dims(tf.repeat(tf.range(batch_size), seq_len), axis=-1), tf.expand_dims(tf.reshape(input_ids, [-1]), axis=-1)), axis=1)
    token_penalties = tf.tensor_scatter_nd_update(token_penalties, indices=indexable_prev_input_ids, updates=tf.reshape(logit_penalties, [-1]))
    return token_penalties