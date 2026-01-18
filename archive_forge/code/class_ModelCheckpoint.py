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
class ModelCheckpoint(Callback):
    """Callback to save the Keras model or model weights at some frequency.

  `ModelCheckpoint` callback is used in conjunction with training using
  `model.fit()` to save a model or weights (in a checkpoint file) at some
  interval, so the model or weights can be loaded later to continue the training
  from the state saved.

  A few options this callback provides include:

  - Whether to only keep the model that has achieved the "best performance" so
    far, or whether to save the model at the end of every epoch regardless of
    performance.
  - Definition of 'best'; which quantity to monitor and whether it should be
    maximized or minimized.
  - The frequency it should save at. Currently, the callback supports saving at
    the end of every epoch, or after a fixed number of training batches.
  - Whether only weights are saved, or the whole model is saved.

  Note: If you get `WARNING:tensorflow:Can save best model only with <name>
  available, skipping` see the description of the `monitor` argument for
  details on how to get this right.

  Example:

  ```python
  model.compile(loss=..., optimizer=...,
                metrics=['accuracy'])

  EPOCHS = 10
  checkpoint_filepath = '/tmp/checkpoint'
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      monitor='val_accuracy',
      mode='max',
      save_best_only=True)

  # Model weights are saved at the end of every epoch, if it's the best seen
  # so far.
  model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])

  # The model weights (that are considered the best) are loaded into the model.
  model.load_weights(checkpoint_filepath)
  ```

  Args:
      filepath: string or `PathLike`, path to save the model file. e.g.
        filepath = os.path.join(working_dir, 'ckpt', file_name). `filepath`
        can contain named formatting options, which will be filled the value of
        `epoch` and keys in `logs` (passed in `on_epoch_end`). For example: if
        `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model
        checkpoints will be saved with the epoch number and the validation loss
        in the filename. The directory of the filepath should not be reused by
        any other callbacks to avoid conflicts.
      monitor: The metric name to monitor. Typically the metrics are set by the
        `Model.compile` method. Note:

        * Prefix the name with `"val_`" to monitor validation metrics.
        * Use `"loss"` or "`val_loss`" to monitor the model's total loss.
        * If you specify metrics as strings, like `"accuracy"`, pass the same
          string (with or without the `"val_"` prefix).
        * If you pass `metrics.Metric` objects, `monitor` should be set to
          `metric.name`
        * If you're not sure about the metric names you can check the contents
          of the `history.history` dictionary returned by
          `history = model.fit()`
        * Multi-output models set additional prefixes on the metric names.

      verbose: verbosity mode, 0 or 1.
      save_best_only: if `save_best_only=True`, it only saves when the model
        is considered the "best" and the latest best model according to the
        quantity monitored will not be overwritten. If `filepath` doesn't
        contain formatting options like `{epoch}` then `filepath` will be
        overwritten by each new better model.
      mode: one of {'auto', 'min', 'max'}. If `save_best_only=True`, the
        decision to overwrite the current save file is made based on either
        the maximization or the minimization of the monitored quantity.
        For `val_acc`, this should be `max`, for `val_loss` this should be
        `min`, etc. In `auto` mode, the mode is set to `max` if the quantities
        monitored are 'acc' or start with 'fmeasure' and are set to `min` for
        the rest of the quantities.
      save_weights_only: if True, then only the model's weights will be saved
        (`model.save_weights(filepath)`), else the full model is saved
        (`model.save(filepath)`).
      save_freq: `'epoch'` or integer. When using `'epoch'`, the callback saves
        the model after each epoch. When using integer, the callback saves the
        model at end of this many batches. If the `Model` is compiled with
        `steps_per_execution=N`, then the saving criteria will be
        checked every Nth batch. Note that if the saving isn't aligned to
        epochs, the monitored metric may potentially be less reliable (it
        could reflect as little as 1 batch, since the metrics get reset every
        epoch). Defaults to `'epoch'`.
      options: Optional `tf.train.CheckpointOptions` object if
        `save_weights_only` is true or optional `tf.saved_model.SaveOptions`
        object if `save_weights_only` is false.
      **kwargs: Additional arguments for backwards compatibility. Possible key
        is `period`.
  """

    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch', options=None, **kwargs):
        super(ModelCheckpoint, self).__init__()
        self._supports_tf_logs = True
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = path_to_string(filepath)
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.epochs_since_last_save = 0
        self._batches_seen_since_last_saving = 0
        self._last_batch_seen = 0
        if save_weights_only:
            if options is None or isinstance(options, checkpoint_options_lib.CheckpointOptions):
                self._options = options or checkpoint_options_lib.CheckpointOptions()
            else:
                raise TypeError('If save_weights_only is True, then `options` must be either None or a tf.train.CheckpointOptions')
        elif options is None or isinstance(options, save_options_lib.SaveOptions):
            self._options = options or save_options_lib.SaveOptions()
        else:
            raise TypeError('If save_weights_only is False, then `options` must beeither None or a tf.saved_model.SaveOptions')
        if 'load_weights_on_restart' in kwargs:
            self.load_weights_on_restart = kwargs['load_weights_on_restart']
            logging.warning('`load_weights_on_restart` argument is deprecated. Please use `model.load_weights()` for loading weights before the start of `model.fit()`.')
        else:
            self.load_weights_on_restart = False
        if 'period' in kwargs:
            self.period = kwargs['period']
            logging.warning('`period` argument is deprecated. Please use `save_freq` to specify the frequency in number of batches seen.')
        else:
            self.period = 1
        if mode not in ['auto', 'min', 'max']:
            logging.warning('ModelCheckpoint mode %s is unknown, fallback to auto mode.', mode)
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        elif 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            self.monitor_op = np.less
            self.best = np.Inf
        if self.save_freq != 'epoch' and (not isinstance(self.save_freq, int)):
            raise ValueError('Unrecognized save_freq: {}'.format(self.save_freq))
        self._chief_worker_only = False

    def on_train_begin(self, logs=None):
        if self.load_weights_on_restart:
            filepath_to_load = self._get_most_recently_modified_file_matching_pattern(self.filepath)
            if filepath_to_load is not None and self._checkpoint_exists(filepath_to_load):
                try:
                    self.model.load_weights(filepath_to_load)
                except (IOError, ValueError) as e:
                    raise ValueError('Error loading file from {}. Reason: {}'.format(filepath_to_load, e))

    def _implements_train_batch_hooks(self):
        return self.save_freq != 'epoch'

    def on_train_batch_end(self, batch, logs=None):
        if self._should_save_on_batch(batch):
            self._save_model(epoch=self._current_epoch, logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.save_freq == 'epoch':
            self._save_model(epoch=epoch, logs=logs)

    def _should_save_on_batch(self, batch):
        """Handles batch-level saving logic, supports steps_per_execution."""
        if self.save_freq == 'epoch':
            return False
        if batch <= self._last_batch_seen:
            add_batches = batch + 1
        else:
            add_batches = batch - self._last_batch_seen
        self._batches_seen_since_last_saving += add_batches
        self._last_batch_seen = batch
        if self._batches_seen_since_last_saving >= self.save_freq:
            self._batches_seen_since_last_saving = 0
            return True
        return False

    def _save_model(self, epoch, logs):
        """Saves the model.

    Args:
        epoch: the epoch this iteration is in.
        logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
    """
        logs = logs or {}
        if isinstance(self.save_freq, int) or self.epochs_since_last_save >= self.period:
            logs = tf_utils.sync_to_numpy_or_python_type(logs)
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, logs)
            try:
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        logging.warning('Can save best model only with %s available, skipping.', self.monitor)
                    elif self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f, saving model to %s' % (epoch + 1, self.monitor, self.best, current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True, options=self._options)
                        else:
                            self.model.save(filepath, overwrite=True, options=self._options)
                    elif self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve from %0.5f' % (epoch + 1, self.monitor, self.best))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True, options=self._options)
                    else:
                        self.model.save(filepath, overwrite=True, options=self._options)
                self._maybe_remove_file()
            except IsADirectoryError as e:
                raise IOError('Please specify a non-directory filepath for ModelCheckpoint. Filepath used is an existing directory: {}'.format(filepath))
            except IOError as e:
                if 'is a directory' in str(e.args[0]).lower():
                    raise IOError('Please specify a non-directory filepath for ModelCheckpoint. Filepath used is an existing directory: {}'.format(filepath))
                raise e

    def _get_file_path(self, epoch, logs):
        """Returns the file path for checkpoint."""
        try:
            file_path = self.filepath.format(epoch=epoch + 1, **logs)
        except KeyError as e:
            raise KeyError('Failed to format this callback filepath: "{}". Reason: {}'.format(self.filepath, e))
        self._write_filepath = distributed_file_utils.write_filepath(file_path, self.model.distribute_strategy)
        return self._write_filepath

    def _maybe_remove_file(self):
        distributed_file_utils.remove_temp_dir_with_filepath(self._write_filepath, self.model.distribute_strategy)

    def _checkpoint_exists(self, filepath):
        """Returns whether the checkpoint `filepath` refers to exists."""
        if filepath.endswith('.h5'):
            return file_io.file_exists_v2(filepath)
        tf_saved_model_exists = file_io.file_exists_v2(filepath)
        tf_weights_only_checkpoint_exists = file_io.file_exists_v2(filepath + '.index')
        return tf_saved_model_exists or tf_weights_only_checkpoint_exists

    def _get_most_recently_modified_file_matching_pattern(self, pattern):
        """Returns the most recently modified filepath matching pattern.

    Pattern may contain python formatting placeholder. If
    `tf.train.latest_checkpoint()` does not return None, use that; otherwise,
    check for most recently modified one that matches the pattern.

    In the rare case where there are more than one pattern-matching file having
    the same modified time that is most recent among all, return the filepath
    that is largest (by `>` operator, lexicographically using the numeric
    equivalents). This provides a tie-breaker when multiple files are most
    recent. Note that a larger `filepath` can sometimes indicate a later time of
    modification (for instance, when epoch/batch is used as formatting option),
    but not necessarily (when accuracy or loss is used). The tie-breaker is
    put in the logic as best effort to return the most recent, and to avoid
    undeterministic result.

    Modified time of a file is obtained with `os.path.getmtime()`.

    This utility function is best demonstrated via an example:

    ```python
    file_pattern = 'f.batch{batch:02d}epoch{epoch:02d}.h5'
    test_dir = self.get_temp_dir()
    path_pattern = os.path.join(test_dir, file_pattern)
    file_paths = [
        os.path.join(test_dir, file_name) for file_name in
        ['f.batch03epoch02.h5', 'f.batch02epoch02.h5', 'f.batch01epoch01.h5']
    ]
    for file_path in file_paths:
      # Write something to each of the files
    self.assertEqual(
        _get_most_recently_modified_file_matching_pattern(path_pattern),
        file_paths[-1])
    ```

    Args:
        pattern: The file pattern that may optionally contain python placeholder
            such as `{epoch:02d}`.

    Returns:
        The most recently modified file's full filepath matching `pattern`. If
        `pattern` does not contain any placeholder, this returns the filepath
        that
        exactly matches `pattern`. Returns `None` if no match is found.
    """
        dir_name = os.path.dirname(pattern)
        base_name = os.path.basename(pattern)
        base_name_regex = '^' + re.sub('{.*}', '.*', base_name) + '$'
        latest_tf_checkpoint = checkpoint_management.latest_checkpoint(dir_name)
        if latest_tf_checkpoint is not None and re.match(base_name_regex, os.path.basename(latest_tf_checkpoint)):
            return latest_tf_checkpoint
        latest_mod_time = 0
        file_path_with_latest_mod_time = None
        n_file_with_latest_mod_time = 0
        file_path_with_largest_file_name = None
        if file_io.file_exists_v2(dir_name):
            for file_name in os.listdir(dir_name):
                if re.match(base_name_regex, file_name):
                    file_path = os.path.join(dir_name, file_name)
                    mod_time = os.path.getmtime(file_path)
                    if file_path_with_largest_file_name is None or file_path > file_path_with_largest_file_name:
                        file_path_with_largest_file_name = file_path
                    if mod_time > latest_mod_time:
                        latest_mod_time = mod_time
                        file_path_with_latest_mod_time = file_path
                        n_file_with_latest_mod_time = 1
                    elif mod_time == latest_mod_time:
                        n_file_with_latest_mod_time += 1
        if n_file_with_latest_mod_time == 1:
            return file_path_with_latest_mod_time
        else:
            return file_path_with_largest_file_name