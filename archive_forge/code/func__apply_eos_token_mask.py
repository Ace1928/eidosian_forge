import inspect
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from ..tf_utils import stable_softmax
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
def _apply_eos_token_mask(self, scores: tf.Tensor) -> tf.Tensor:
    eos_token_id_mask = tf.range(scores.shape[-1]) == self.eos_token_id
    scores = tf.where(eos_token_id_mask, float('-inf'), scores)
    return scores