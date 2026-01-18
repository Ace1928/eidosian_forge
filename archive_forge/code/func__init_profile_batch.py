import collections
import copy
import csv
import json
import os
import re
import sys
import time
import numpy as np
from tensorflow.core.framework import summary_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import checkpoint_options as checkpoint_options_lib
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.keras import backend
from tensorflow.python.keras.distribute import distributed_file_utils
from tensorflow.python.keras.distribute import worker_training_state
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.io_utils import path_to_string
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.saved_model import save_options as save_options_lib
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
def _init_profile_batch(self, profile_batch):
    """Validate profile_batch value and set the range of batches to profile.
    Sets values of _start_batch and _stop_batch attributes,
    specifying the start and stop batch to profile.
    Setting `profile_batch=0` disables profiling.

    Args:
      profile_batch: The range of batches to profile. Should be a non-negative
        integer or a comma separated string of pair of positive integers. A pair
        of positive integers signify a range of batches to profile.

    Raises:
      ValueError: If profile_batch is not an integer or a comma separated pair
                  of positive integers.

    """
    profile_batch_error_message = 'profile_batch must be a non-negative integer or 2-tuple of positive integers. A pair of positive integers signifies a range of batches to profile. Found: {}'.format(profile_batch)
    if isinstance(profile_batch, str):
        profile_batch = str(profile_batch).split(',')
        profile_batch = nest.map_structure(int, profile_batch)
    if isinstance(profile_batch, int):
        self._start_batch = profile_batch
        self._stop_batch = profile_batch
    elif isinstance(profile_batch, (tuple, list)) and len(profile_batch) == 2:
        self._start_batch, self._stop_batch = profile_batch
    else:
        raise ValueError(profile_batch_error_message)
    if self._start_batch < 0 or self._stop_batch < self._start_batch:
        raise ValueError(profile_batch_error_message)
    self._profiler_started = False
    if self._start_batch > 0:
        self._start_profiler(logdir='')
        self._stop_profiler(save=False)
    self._is_tracing = False
    self._should_trace = not (self._start_batch == 0 and self._stop_batch == 0)