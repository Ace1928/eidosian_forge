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
class Callback:
    """Abstract base class used to build new callbacks.

  Callbacks can be passed to keras methods such as `fit`, `evaluate`, and
  `predict` in order to hook into the various stages of the model training and
  inference lifecycle.

  To create a custom callback, subclass `keras.callbacks.Callback` and override
  the method associated with the stage of interest. See
  https://www.tensorflow.org/guide/keras/custom_callback for more information.

  Example:

  >>> training_finished = False
  >>> class MyCallback(tf.keras.callbacks.Callback):
  ...   def on_train_end(self, logs=None):
  ...     global training_finished
  ...     training_finished = True
  >>> model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
  >>> model.compile(loss='mean_squared_error')
  >>> model.fit(tf.constant([[1.0]]), tf.constant([[1.0]]),
  ...           callbacks=[MyCallback()])
  >>> assert training_finished == True

  If you want to use `Callback` objects in a custom training loop:

  1. You should pack all your callbacks into a single `callbacks.CallbackList`
     so they can all be called together.
  2. You will need to manually call all the `on_*` methods at the apropriate
     locations in your loop. Like this:

     ```
     callbacks =  tf.keras.callbacks.CallbackList([...])
     callbacks.append(...)

     callbacks.on_train_begin(...)
     for epoch in range(EPOCHS):
       callbacks.on_epoch_begin(epoch)
       for i, data in dataset.enumerate():
         callbacks.on_train_batch_begin(i)
         batch_logs = model.train_step(data)
         callbacks.on_train_batch_end(i, batch_logs)
       epoch_logs = ...
       callbacks.on_epoch_end(epoch, epoch_logs)
     final_logs=...
     callbacks.on_train_end(final_logs)
     ```

  Attributes:
      params: Dict. Training parameters
          (eg. verbosity, batch size, number of epochs...).
      model: Instance of `keras.models.Model`.
          Reference of the model being trained.

  The `logs` dictionary that callback methods
  take as argument will contain keys for quantities relevant to
  the current batch or epoch (see method-specific docstrings).
  """

    def __init__(self):
        self.validation_data = None
        self.model = None
        self._chief_worker_only = None
        self._supports_tf_logs = False

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_batch_begin(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_begin`."""

    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_batch_end(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_end`."""

    @doc_controls.for_subclass_implementers
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.

    Subclasses should override for any actions to run. This function should only
    be called during TRAIN mode.

    Args:
        epoch: Integer, index of epoch.
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

    @doc_controls.for_subclass_implementers
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.

    Subclasses should override for any actions to run. This function should only
    be called during TRAIN mode.

    Args:
        epoch: Integer, index of epoch.
        logs: Dict, metric results for this training epoch, and for the
          validation epoch if validation is performed. Validation result keys
          are prefixed with `val_`. For training epoch, the values of the
         `Model`'s metrics are returned. Example : `{'loss': 0.2, 'accuracy':
           0.7}`.
    """

    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_train_batch_begin(self, batch, logs=None):
        """Called at the beginning of a training batch in `fit` methods.

    Subclasses should override for any actions to run.

    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict, contains the return value of `model.train_step`. Typically,
          the values of the `Model`'s metrics are returned.  Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
    """
        self.on_batch_begin(batch, logs=logs)

    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch in `fit` methods.

    Subclasses should override for any actions to run.

    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """
        self.on_batch_end(batch, logs=logs)

    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_test_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `evaluate` methods.

    Also called at the beginning of a validation batch in the `fit`
    methods, if validation data is provided.

    Subclasses should override for any actions to run.

    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict, contains the return value of `model.test_step`. Typically,
          the values of the `Model`'s metrics are returned.  Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
    """

    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_test_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `evaluate` methods.

    Also called at the end of a validation batch in the `fit`
    methods, if validation data is provided.

    Subclasses should override for any actions to run.

    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """

    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_predict_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `predict` methods.

    Subclasses should override for any actions to run.

    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict, contains the return value of `model.predict_step`,
          it typically returns a dict with a key 'outputs' containing
          the model's outputs.
    """

    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_predict_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `predict` methods.

    Subclasses should override for any actions to run.

    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """

    @doc_controls.for_subclass_implementers
    def on_train_begin(self, logs=None):
        """Called at the beginning of training.

    Subclasses should override for any actions to run.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

    @doc_controls.for_subclass_implementers
    def on_train_end(self, logs=None):
        """Called at the end of training.

    Subclasses should override for any actions to run.

    Args:
        logs: Dict. Currently the output of the last call to `on_epoch_end()`
          is passed to this argument for this method but that may change in
          the future.
    """

    @doc_controls.for_subclass_implementers
    def on_test_begin(self, logs=None):
        """Called at the beginning of evaluation or validation.

    Subclasses should override for any actions to run.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

    @doc_controls.for_subclass_implementers
    def on_test_end(self, logs=None):
        """Called at the end of evaluation or validation.

    Subclasses should override for any actions to run.

    Args:
        logs: Dict. Currently the output of the last call to
          `on_test_batch_end()` is passed to this argument for this method
          but that may change in the future.
    """

    @doc_controls.for_subclass_implementers
    def on_predict_begin(self, logs=None):
        """Called at the beginning of prediction.

    Subclasses should override for any actions to run.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

    @doc_controls.for_subclass_implementers
    def on_predict_end(self, logs=None):
        """Called at the end of prediction.

    Subclasses should override for any actions to run.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

    def _implements_train_batch_hooks(self):
        """Determines if this Callback should be called for each train batch."""
        return not generic_utils.is_default(self.on_batch_begin) or not generic_utils.is_default(self.on_batch_end) or (not generic_utils.is_default(self.on_train_batch_begin)) or (not generic_utils.is_default(self.on_train_batch_end))

    def _implements_test_batch_hooks(self):
        """Determines if this Callback should be called for each test batch."""
        return not generic_utils.is_default(self.on_test_batch_begin) or not generic_utils.is_default(self.on_test_batch_end)

    def _implements_predict_batch_hooks(self):
        """Determines if this Callback should be called for each predict batch."""
        return not generic_utils.is_default(self.on_predict_batch_begin) or not generic_utils.is_default(self.on_predict_batch_end)