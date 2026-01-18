import abc
import contextlib
import functools
import itertools
import math
import random
import numpy as np
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import dataset_creator
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
class DataHandler(object):
    """Handles iterating over epoch-level `tf.data.Iterator` objects."""

    def __init__(self, x, y=None, sample_weight=None, batch_size=None, steps_per_epoch=None, initial_epoch=0, epochs=1, shuffle=False, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, model=None, steps_per_execution=None, distribute=True):
        """Initializes a `DataHandler`.

    Arguments:
      x: See `Model.fit`.
      y: See `Model.fit`.
      sample_weight: See `Model.fit`.
      batch_size: See `Model.fit`.
      steps_per_epoch: See `Model.fit`.
      initial_epoch: See `Model.fit`.
      epochs: See `Model.fit`.
      shuffle: See `Model.fit`.
      class_weight: See `Model.fit`.
      max_queue_size: See `Model.fit`.
      workers: See `Model.fit`.
      use_multiprocessing: See `Model.fit`.
      model: The `Model` instance. Needed in order to correctly `build` the
        `Model` using generator-like inputs (see `GeneratorDataAdapter`).
      steps_per_execution: See `Model.compile`.
      distribute: Whether to distribute the `tf.dataset`.
        `PreprocessingLayer.adapt` does not support distributed datasets,
        `Model` should always set this to `True`.
    """
        self._initial_epoch = initial_epoch
        self._epochs = epochs
        self._insufficient_data = False
        self._model = model
        if steps_per_execution is None:
            self._steps_per_execution = 1
            self._steps_per_execution_value = 1
        else:
            self._steps_per_execution = steps_per_execution
            self._steps_per_execution_value = steps_per_execution.numpy().item()
        adapter_cls = select_data_adapter(x, y)
        self._adapter = adapter_cls(x, y, batch_size=batch_size, steps=steps_per_epoch, epochs=epochs - initial_epoch, sample_weights=sample_weight, shuffle=shuffle, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing, distribution_strategy=distribute_lib.get_strategy(), model=model)
        strategy = distribute_lib.get_strategy()
        self._current_step = 0
        self._step_increment = self._steps_per_execution_value - 1
        self._insufficient_data = False
        self._configure_dataset_and_inferred_steps(strategy, x, steps_per_epoch, class_weight, distribute)

    def _configure_dataset_and_inferred_steps(self, strategy, x, steps_per_epoch, class_weight, distribute):
        """Configure the `_dataset` and `_inferred_steps` attributes."""
        del x
        dataset = self._adapter.get_dataset()
        if class_weight:
            dataset = dataset.map(_make_class_weight_map_fn(class_weight))
        self._inferred_steps = self._infer_steps(steps_per_epoch, dataset)
        if distribute and (not _is_distributed_dataset(dataset)):
            dataset = strategy.experimental_distribute_dataset(dataset)
        self._dataset = dataset
        self._validate_data_handler()

    def enumerate_epochs(self):
        """Yields `(epoch, tf.data.Iterator)`."""
        with self._truncate_execution_to_epoch():
            data_iterator = iter(self._dataset)
            for epoch in range(self._initial_epoch, self._epochs):
                if self._insufficient_data:
                    break
                if self._adapter.should_recreate_iterator():
                    data_iterator = iter(self._dataset)
                yield (epoch, data_iterator)
                self._adapter.on_epoch_end()

    @contextlib.contextmanager
    def _truncate_execution_to_epoch(self):
        """Truncates steps per execution to at most one epoch."""
        should_truncate = self._inferred_steps is not None and self._steps_per_execution_value > self._inferred_steps
        original_value = self._steps_per_execution_value
        try:
            if should_truncate:
                self._steps_per_execution.assign(self._inferred_steps)
                self._steps_per_execution_value = self._inferred_steps
            yield
        finally:
            if should_truncate:
                self._steps_per_execution.assign(original_value)
                self._steps_per_execution_value = original_value

    def sync(self):
        context.async_wait()

    @contextlib.contextmanager
    def catch_stop_iteration(self):
        """Catches errors when an iterator runs out of data."""
        try:
            yield
            self.sync()
        except (StopIteration, errors.OutOfRangeError):
            if self._inferred_steps is None:
                self._inferred_steps = self._current_step
            else:
                self._insufficient_data = True
                total_epochs = self._epochs - self._initial_epoch
                logging.warning('Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, {} batches). You may need to use the repeat() function when building your dataset.'.format(total_epochs * self._inferred_steps))

    def steps(self):
        """Yields steps for the current epoch."""
        self._current_step = 0
        while self._inferred_steps is None or self._current_step < self._inferred_steps:
            if self._insufficient_data:
                break
            can_run_full_execution = self._steps_per_execution_value == 1 or self._inferred_steps is None or self._inferred_steps - self._current_step >= self._steps_per_execution_value
            if can_run_full_execution:
                self._step_increment = self._steps_per_execution_value - 1
                yield self._current_step
                self._current_step += self._steps_per_execution_value
            else:
                steps_remaining = self._inferred_steps - self._current_step
                self._steps_per_execution.assign(steps_remaining)
                self._step_increment = steps_remaining - 1
                yield self._current_step
                self._current_step += steps_remaining
                self._steps_per_execution.assign(self._steps_per_execution_value)

    @property
    def step_increment(self):
        """The number to increment the step for `on_batch_end` methods."""
        return self._step_increment

    @property
    def inferred_steps(self):
        """The inferred steps per epoch of the created `Dataset`.

    This will be `None` in the case where:

    (1) A `Dataset` of unknown cardinality was passed to the `DataHandler`, and
    (2) `steps_per_epoch` was not provided, and
    (3) The first epoch of iteration has not yet completed.

    Returns:
      The inferred steps per epoch of the created `Dataset`.
    """
        return self._inferred_steps

    @property
    def should_sync(self):
        return self._inferred_steps is None

    def _log_indefinite_training_warning(self):
        logging.warning('The training loop will run indefinitely since you have set `steps_per_epoch=-1`. Please use batch-level callbacks to save checkpoints or log training progress, etc')

    def _infer_steps(self, steps, dataset):
        """Infers steps_per_epoch needed to loop through a dataset."""
        if steps == -1:
            self._log_indefinite_training_warning()
            return None
        if steps is not None:
            return steps
        adapter_steps = self._adapter.get_size()
        if adapter_steps is not None:
            return adapter_steps
        size = cardinality.cardinality(dataset)
        if size == cardinality.INFINITE and steps is None:
            raise ValueError('When passing an infinitely repeating dataset, please specify a `steps_per_epoch` value so that epoch level callbacks continue to work. The value can be arbitrary, or a number that you think correctly defines the size of an epoch. Epoch-level callbacks will then be called at this interval.')
        if size >= 0:
            return size.numpy().item()
        return None

    @property
    def _samples(self):
        return self._adapter.get_samples()

    def _validate_data_handler(self):
        if self._steps_per_execution_value > 1 and self._inferred_steps is None:
            raise ValueError('Could not infer the size of the data. With `steps_per_execution > 1`, you must specify the number of steps to run.')