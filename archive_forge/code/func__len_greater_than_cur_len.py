import inspect
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from ..tf_utils import stable_softmax
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
def _len_greater_than_cur_len():
    return tf.cond(tf.math.greater(self.bad_word_seqs_len[bad_word_seq_number], tf.shape(row_input_ids)[0]), lambda: tf.zeros((), dtype=tf.bool), _match_found)