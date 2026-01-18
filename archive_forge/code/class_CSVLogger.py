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
class CSVLogger(Callback):
    """Callback that streams epoch results to a CSV file.

  Supports all values that can be represented as a string,
  including 1D iterables such as `np.ndarray`.

  Example:

  ```python
  csv_logger = CSVLogger('training.log')
  model.fit(X_train, Y_train, callbacks=[csv_logger])
  ```

  Args:
      filename: Filename of the CSV file, e.g. `'run/log.csv'`.
      separator: String used to separate elements in the CSV file.
      append: Boolean. True: append if file exists (useful for continuing
          training). False: overwrite existing file.
  """

    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = path_to_string(filename)
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        super(CSVLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if file_io.file_exists_v2(self.filename):
                with gfile.GFile(self.filename, 'r') as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'
        self.csv_file = gfile.GFile(self.filename, mode)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif isinstance(k, collections.abc.Iterable) and (not is_zero_dim_ndarray):
                return '"[%s]"' % ', '.join(map(str, k))
            else:
                return k
        if self.keys is None:
            self.keys = sorted(logs.keys())
        if self.model.stop_training:
            logs = dict(((k, logs[k]) if k in logs else (k, 'NA') for k in self.keys))
        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.sep
            fieldnames = ['epoch'] + self.keys
            self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()
        row_dict = collections.OrderedDict({'epoch': epoch})
        row_dict.update(((key, handle_value(logs[key])) for key in self.keys))
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None