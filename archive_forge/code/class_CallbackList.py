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
class CallbackList:
    """Container abstracting a list of callbacks."""

    def __init__(self, callbacks=None, add_history=False, add_progbar=False, model=None, **params):
        """Container for `Callback` instances.

    This object wraps a list of `Callback` instances, making it possible
    to call them all at once via a single endpoint
    (e.g. `callback_list.on_epoch_end(...)`).

    Args:
      callbacks: List of `Callback` instances.
      add_history: Whether a `History` callback should be added, if one does not
        already exist in the `callbacks` list.
      add_progbar: Whether a `ProgbarLogger` callback should be added, if one
        does not already exist in the `callbacks` list.
      model: The `Model` these callbacks are used with.
      **params: If provided, parameters will be passed to each `Callback` via
        `Callback.set_params`.
    """
        self.callbacks = nest.flatten(callbacks) if callbacks else []
        self._add_default_callbacks(add_history, add_progbar)
        if model:
            self.set_model(model)
        if params:
            self.set_params(params)
        self._supports_tf_logs = all((getattr(cb, '_supports_tf_logs', False) for cb in self.callbacks))
        self._batch_hooks_support_tf_logs = all((getattr(cb, '_supports_tf_logs', False) for cb in self.callbacks if cb._implements_train_batch_hooks() or cb._implements_test_batch_hooks() or cb._implements_predict_batch_hooks()))
        self._should_call_train_batch_hooks = any((cb._implements_train_batch_hooks() for cb in self.callbacks))
        self._should_call_test_batch_hooks = any((cb._implements_test_batch_hooks() for cb in self.callbacks))
        self._should_call_predict_batch_hooks = any((cb._implements_predict_batch_hooks() for cb in self.callbacks))
        self._disallow_batch_hooks_in_ps_strategy()
        self._check_timing = any((cbk.__class__.__name__ not in globals() for cbk in self.callbacks))
        self._num_batches_for_timing_check = 5
        self._hook_times = {}
        self._batch_start_time = None
        self._batch_times = []

    def _add_default_callbacks(self, add_history, add_progbar):
        """Adds `Callback`s that are always present."""
        self._progbar = None
        self._history = None
        for cb in self.callbacks:
            if isinstance(cb, ProgbarLogger):
                self._progbar = cb
            elif isinstance(cb, History):
                self._history = cb
        if self._progbar is None and add_progbar:
            self._progbar = ProgbarLogger(count_mode='steps')
            self.callbacks.insert(0, self._progbar)
        if self._history is None and add_history:
            self._history = History()
            self.callbacks.append(self._history)

    def _process_logs(self, logs, is_batch_hook=False):
        """Turns tensors into numpy arrays or Python scalars if necessary."""
        if logs is None:
            return {}
        if self._supports_tf_logs:
            return logs
        if is_batch_hook and self._batch_hooks_support_tf_logs:
            return logs
        return tf_utils.sync_to_numpy_or_python_type(logs)

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        self.model = model
        if self._history:
            model.history = self._history
        for callback in self.callbacks:
            callback.set_model(model)

    def _call_batch_hook(self, mode, hook, batch, logs=None):
        """Helper function for all batch_{begin | end} methods."""
        if not self.callbacks:
            return
        if hook == 'begin':
            self._call_batch_begin_hook(mode, batch, logs)
        elif hook == 'end':
            self._call_batch_end_hook(mode, batch, logs)
        else:
            raise ValueError('Unrecognized hook: {}'.format(hook))

    def _call_batch_begin_hook(self, mode, batch, logs):
        """Helper function for `on_*_batch_begin` methods."""
        hook_name = 'on_{mode}_batch_begin'.format(mode=mode)
        self._call_batch_hook_helper(hook_name, batch, logs)
        if self._check_timing:
            self._batch_start_time = time.time()

    def _call_batch_end_hook(self, mode, batch, logs):
        """Helper function for `on_*_batch_end` methods."""
        hook_name = 'on_{mode}_batch_end'.format(mode=mode)
        if self._check_timing and batch >= 1:
            batch_time = time.time() - self._batch_start_time
            self._batch_times.append(batch_time)
        self._call_batch_hook_helper(hook_name, batch, logs)
        if len(self._batch_times) >= self._num_batches_for_timing_check:
            end_hook_name = hook_name
            begin_hook_name = 'on_{mode}_batch_begin'.format(mode=mode)
            avg_batch_time = sum(self._batch_times) / len(self._batch_times)
            avg_end_hook_time = sum(self._hook_times[end_hook_name]) / len(self._hook_times[end_hook_name])
            avg_begin_hook_time = sum(self._hook_times[begin_hook_name]) / len(self._hook_times[begin_hook_name])
            threshold_time = 1.0 * avg_batch_time
            warning_msg = 'Callback method `{hook}` is slow compared to the batch time (batch time: {batch_time:.4f}s vs `{hook}` time: {hook_time:.4f}s). Check your callbacks.'
            if avg_begin_hook_time > threshold_time:
                logging.warning(warning_msg.format(hook=begin_hook_name, batch_time=avg_batch_time, hook_time=avg_begin_hook_time))
            if avg_end_hook_time > threshold_time:
                logging.warning(warning_msg.format(hook=end_hook_name, batch_time=avg_batch_time, hook_time=avg_end_hook_time))
            self._check_timing = False
            self._batch_start_time = None
            self._batch_times = []
            self._hook_times = {}

    def _call_batch_hook_helper(self, hook_name, batch, logs):
        """Helper function for `on_*_batch_*` methods."""
        if self._check_timing:
            start_time = time.time()
        logs = self._process_logs(logs, is_batch_hook=True)
        for callback in self.callbacks:
            hook = getattr(callback, hook_name)
            hook(batch, logs)
        if self._check_timing:
            if hook_name not in self._hook_times:
                self._hook_times[hook_name] = []
            self._hook_times[hook_name].append(time.time() - start_time)

    def _call_begin_hook(self, mode):
        """Helper function for on_{train|test|predict}_begin methods."""
        if mode == ModeKeys.TRAIN:
            self.on_train_begin()
        elif mode == ModeKeys.TEST:
            self.on_test_begin()
        else:
            self.on_predict_begin()

    def _call_end_hook(self, mode):
        """Helper function for on_{train|test|predict}_end methods."""
        if mode == ModeKeys.TRAIN:
            self.on_train_end()
        elif mode == ModeKeys.TEST:
            self.on_test_end()
        else:
            self.on_predict_end()

    def on_batch_begin(self, batch, logs=None):
        if self._should_call_train_batch_hooks:
            self._call_batch_hook(ModeKeys.TRAIN, 'begin', batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        if self._should_call_train_batch_hooks:
            self._call_batch_hook(ModeKeys.TRAIN, 'end', batch, logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        """Calls the `on_epoch_begin` methods of its callbacks.

    This function should only be called during TRAIN mode.

    Args:
        epoch: Integer, index of epoch.
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """Calls the `on_epoch_end` methods of its callbacks.

    This function should only be called during TRAIN mode.

    Args:
        epoch: Integer, index of epoch.
        logs: Dict, metric results for this training epoch, and for the
          validation epoch if validation is performed. Validation result keys
          are prefixed with `val_`.
    """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_train_batch_begin(self, batch, logs=None):
        """Calls the `on_train_batch_begin` methods of its callbacks.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict, contains the return value of `model.train_step`. Typically,
          the values of the `Model`'s metrics are returned.  Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
    """
        if self._should_call_train_batch_hooks:
            self._call_batch_hook(ModeKeys.TRAIN, 'begin', batch, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        """Calls the `on_train_batch_end` methods of its callbacks.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """
        if self._should_call_train_batch_hooks:
            self._call_batch_hook(ModeKeys.TRAIN, 'end', batch, logs=logs)

    def on_test_batch_begin(self, batch, logs=None):
        """Calls the `on_test_batch_begin` methods of its callbacks.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict, contains the return value of `model.test_step`. Typically,
          the values of the `Model`'s metrics are returned.  Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
    """
        if self._should_call_test_batch_hooks:
            self._call_batch_hook(ModeKeys.TEST, 'begin', batch, logs=logs)

    def on_test_batch_end(self, batch, logs=None):
        """Calls the `on_test_batch_end` methods of its callbacks.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """
        if self._should_call_test_batch_hooks:
            self._call_batch_hook(ModeKeys.TEST, 'end', batch, logs=logs)

    def on_predict_batch_begin(self, batch, logs=None):
        """Calls the `on_predict_batch_begin` methods of its callbacks.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict, contains the return value of `model.predict_step`,
          it typically returns a dict with a key 'outputs' containing
          the model's outputs.
    """
        if self._should_call_predict_batch_hooks:
            self._call_batch_hook(ModeKeys.PREDICT, 'begin', batch, logs=logs)

    def on_predict_batch_end(self, batch, logs=None):
        """Calls the `on_predict_batch_end` methods of its callbacks.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """
        if self._should_call_predict_batch_hooks:
            self._call_batch_hook(ModeKeys.PREDICT, 'end', batch, logs=logs)

    def on_train_begin(self, logs=None):
        """Calls the `on_train_begin` methods of its callbacks.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """Calls the `on_train_end` methods of its callbacks.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_test_begin(self, logs=None):
        """Calls the `on_test_begin` methods of its callbacks.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    def on_test_end(self, logs=None):
        """Calls the `on_test_end` methods of its callbacks.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_test_end(logs)

    def on_predict_begin(self, logs=None):
        """Calls the 'on_predict_begin` methods of its callbacks.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_predict_begin(logs)

    def on_predict_end(self, logs=None):
        """Calls the `on_predict_end` methods of its callbacks.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_predict_end(logs)

    def __iter__(self):
        return iter(self.callbacks)

    def _disallow_batch_hooks_in_ps_strategy(self):
        """Error out if batch-level callbacks are passed with PSStrategy."""
        strategy = distribute_lib.get_strategy()
        if strategy._should_use_with_coordinator:
            unsupported_callbacks = []
            for cb in self.callbacks:
                if getattr(cb, '_supports_tf_logs', False):
                    continue
                if cb._implements_train_batch_hooks() or cb._implements_test_batch_hooks() or cb._implements_predict_batch_hooks():
                    unsupported_callbacks.append(cb)
            if unsupported_callbacks:
                raise ValueError('Batch-level `Callback`s are not supported with `ParameterServerStrategy`. Found unsupported callbacks: {}'.format(unsupported_callbacks))