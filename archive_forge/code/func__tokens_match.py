import inspect
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from ..tf_utils import stable_softmax
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
def _tokens_match(bad_word_seq_number):

    def _len_one():
        return tf.cond(tf.math.equal(self.bad_word_seqs_len[bad_word_seq_number], 1), lambda: tf.ones((), dtype=tf.bool), _len_greater_than_cur_len)

    def _len_greater_than_cur_len():
        return tf.cond(tf.math.greater(self.bad_word_seqs_len[bad_word_seq_number], tf.shape(row_input_ids)[0]), lambda: tf.zeros((), dtype=tf.bool), _match_found)

    def _match_found():
        compare_len = self.bad_word_seqs_len[bad_word_seq_number] - 1
        return tf.cond(tf.math.reduce_all(tf.math.equal(row_input_ids[-compare_len:], self.bad_word_seqs_ids[bad_word_seq_number, :compare_len])), lambda: tf.ones((), dtype=tf.bool), lambda: tf.zeros((), dtype=tf.bool))
    match = _len_one()
    return match