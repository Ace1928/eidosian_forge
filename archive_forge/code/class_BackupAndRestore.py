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
class BackupAndRestore(Callback):
    """Callback to back up and restore the training state.

  `BackupAndRestore` callback is intended to recover from interruptions that
  happened in the middle of a model.fit execution by backing up the
  training states in a temporary checkpoint file (based on TF CheckpointManager)
  at the end of each epoch. If training restarted before completion, the
  training state and model are restored to the most recently saved state at the
  beginning of a new model.fit() run.
  Note that user is responsible to bring jobs back up.
  This callback is important for the backup and restore mechanism for fault
  tolerance purpose. And the model to be restored from an previous checkpoint is
  expected to be the same as the one used to back up. If user changes arguments
  passed to compile or fit, the checkpoint saved for fault tolerance can become
  invalid.

  Note:
  1. This callback is not compatible with disabling eager execution.
  2. A checkpoint is saved at the end of each epoch, when restoring we'll redo
  any partial work from an unfinished epoch in which the training got restarted
  (so the work done before a interruption doesn't affect the final model state).
  3. This works for both single worker and multi-worker mode, only
  MirroredStrategy and MultiWorkerMirroredStrategy are supported for now.

  Example:

  >>> class InterruptingCallback(tf.keras.callbacks.Callback):
  ...   def on_epoch_begin(self, epoch, logs=None):
  ...     if epoch == 4:
  ...       raise RuntimeError('Interrupting!')
  >>> callback = tf.keras.callbacks.experimental.BackupAndRestore(
  ... backup_dir="/tmp/backup")
  >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
  >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
  >>> try:
  ...   model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10,
  ...             batch_size=1, callbacks=[callback, InterruptingCallback()],
  ...             verbose=0)
  ... except:
  ...   pass
  >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10,
  ...             batch_size=1, callbacks=[callback], verbose=0)
  >>> # Only 6 more epochs are run, since first trainning got interrupted at
  >>> # zero-indexed epoch 4, second training will continue from 4 to 9.
  >>> len(history.history['loss'])
  6

  Args:
      backup_dir: String, path to store the checkpoint.
        e.g. backup_dir = os.path.join(working_dir, 'backup')
        This is the directory in which the system stores temporary files to
        recover the model from jobs terminated unexpectedly. The directory
        cannot be reused elsewhere to store other files, e.g. by
        BackupAndRestore callback of another training, or by another callback
        (ModelCheckpoint) of the same training.
  """

    def __init__(self, backup_dir):
        super(BackupAndRestore, self).__init__()
        self.backup_dir = backup_dir
        self._supports_tf_logs = True
        self._supported_strategies = (mirrored_strategy.MirroredStrategy, collective_all_reduce_strategy.CollectiveAllReduceStrategy, tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV2, parameter_server_strategy_v2.ParameterServerStrategyV2)
        if not context.executing_eagerly():
            if ops.inside_function():
                raise ValueError("This Callback's method contains Python state and should be called outside of `tf.function`s.")
            else:
                raise ValueError('BackupAndRestore only supports eager mode. In graph mode, consider using ModelCheckpoint to manually save and restore weights with `model.load_weights()` and by providing `initial_epoch` in `model.fit()` for fault tolerance.')
        self._chief_worker_only = False

    def on_train_begin(self, logs=None):
        if self.model._distribution_strategy and (not isinstance(self.model.distribute_strategy, self._supported_strategies)):
            raise NotImplementedError('%s is not supported yet. Currently BackupAndRestore callback only supports empty strategy, MirroredStrategy, MultiWorkerMirroredStrategy and TPUStrategy.' % type(self.model.distribute_strategy).__name__)
        self.model._training_state = worker_training_state.WorkerTrainingState(self.model, self.backup_dir)
        self._training_state = self.model._training_state
        self._training_state.restore()

    def on_train_end(self, logs=None):
        self._training_state.delete_backup()
        del self._training_state
        del self.model._training_state

    def on_epoch_end(self, epoch, logs=None):
        self._training_state.back_up(epoch)