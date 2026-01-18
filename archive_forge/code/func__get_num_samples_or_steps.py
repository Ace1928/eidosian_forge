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
def _get_num_samples_or_steps(data, steps_per_epoch):
    """Returns number of samples or steps, and whether to use steps count mode."""
    flat_inputs = nest.flatten(data)
    if hasattr(flat_inputs[0], 'shape'):
        return (int(flat_inputs[0].shape[0]), False)
    return (steps_per_epoch, True)