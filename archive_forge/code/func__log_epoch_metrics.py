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
def _log_epoch_metrics(self, epoch, logs):
    """Writes epoch metrics out as scalar summaries.

    Args:
        epoch: Int. The global step to use for TensorBoard.
        logs: Dict. Keys are scalar summary names, values are scalars.
    """
    if not logs:
        return
    train_logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
    val_logs = {k: v for k, v in logs.items() if k.startswith('val_')}
    train_logs = self._collect_learning_rate(train_logs)
    if self.write_steps_per_second:
        train_logs['steps_per_second'] = self._compute_steps_per_second()
    with summary_ops_v2.record_if(True):
        if train_logs:
            with self._train_writer.as_default():
                for name, value in train_logs.items():
                    summary_ops_v2.scalar('epoch_' + name, value, step=epoch)
        if val_logs:
            with self._val_writer.as_default():
                for name, value in val_logs.items():
                    name = name[4:]
                    summary_ops_v2.scalar('epoch_' + name, value, step=epoch)