import functools
import math
import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
def _gen(data):
    """Makes a generator out of a structure of NumPy/EagerTensors."""
    index_array = np.arange(num_samples)
    for _ in range(epochs):
        if shuffle:
            np.random.shuffle(index_array)
        batches = generic_utils.make_batches(num_samples, batch_size)
        for batch_start, batch_end in batches:
            batch_ids = index_array[batch_start:batch_end]
            flat_batch_data = training_utils.slice_arrays(nest.flatten(data), batch_ids, contiguous=not shuffle)
            yield nest.pack_sequence_as(data, flat_batch_data)