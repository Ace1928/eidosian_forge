import json
import random
import numpy as np
from keras.src.utils import data_utils
from tensorflow.python.util.tf_export import keras_export
def _remove_long_seq(maxlen, seq, label):
    """Removes sequences that exceed the maximum length.

    Args:
        maxlen: Int, maximum length of the output sequences.
        seq: List of lists, where each sublist is a sequence.
        label: List where each element is an integer.

    Returns:
        new_seq, new_label: shortened lists for `seq` and `label`.
    """
    new_seq, new_label = ([], [])
    for x, y in zip(seq, label):
        if len(x) < maxlen:
            new_seq.append(x)
            new_label.append(y)
    return (new_seq, new_label)