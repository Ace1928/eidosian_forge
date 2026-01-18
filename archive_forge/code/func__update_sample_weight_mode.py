import functools
import numpy as np
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.distribute import distributed_training_utils_v1
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.utils.generic_utils import make_batches
from tensorflow.python.keras.utils.generic_utils import slice_arrays
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
def _update_sample_weight_mode(model, mode, inputs):
    """Updates the sample_weight_mode of a given model."""
    if mode == ModeKeys.PREDICT:
        return
    sample_weights = None
    if not callable(inputs):
        sample_weights = inputs[len(model._feed_inputs) + len(model._feed_targets):]
        has_learning_phase_pl = mode == ModeKeys.TRAIN and (not isinstance(backend.symbolic_learning_phase(), int))
        if has_learning_phase_pl:
            sample_weights = sample_weights[:-1]
        model._update_sample_weight_modes(sample_weights=sample_weights)
    if model._distribution_strategy:
        distributed_training_utils_v1._update_sample_weight_modes(model, mode, sample_weights)