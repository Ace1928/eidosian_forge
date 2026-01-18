from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import random
import types as tp
import numpy as np
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator.inputs.queues import feeding_queue_runner as fqr
def _fill_array(arr, seq, fillvalue=0):
    """Recursively fills padded arr with elements from seq.

  If length of seq is less than arr padded length, fillvalue used.
  Args:
    arr: Padded tensor of shape [batch_size, ..., max_padded_dim_len].
    seq: Non-padded list of data samples of shape
      [batch_size, ..., padded_dim(None)]
    fillvalue: Default fillvalue to use.
  """
    if arr.ndim == 1:
        try:
            len_ = len(seq)
        except TypeError:
            len_ = 0
        arr[:len_] = seq
        arr[len_:] = fillvalue
    else:
        for subarr, subseq in six.moves.zip_longest(arr, seq, fillvalue=()):
            _fill_array(subarr, subseq, fillvalue)