import abc
import collections
import functools
import glob
import os
import threading
import time
import weakref
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import async_checkpoint_helper
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import graph_view as graph_view_lib
from tensorflow.python.checkpoint import restore as restore_lib
from tensorflow.python.checkpoint import save_util
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import util
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops as io_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import saver as v1_saver_lib
from tensorflow.python.training.saving import saveable_object as saveable_object_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.Checkpoint'])
class CheckpointV1(autotrackable.AutoTrackable):
    """Groups trackable objects, saving and restoring them.

  `Checkpoint`'s constructor accepts keyword arguments whose values are types
  that contain trackable state, such as `tf.compat.v1.train.Optimizer`
  implementations, `tf.Variable`, `tf.keras.Layer` implementations, or
  `tf.keras.Model` implementations. It saves these values with a checkpoint, and
  maintains a `save_counter` for numbering checkpoints.

  Example usage when graph building:

  ```python
  import tensorflow as tf
  import os

  checkpoint_directory = "/tmp/training_checkpoints"
  checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
  train_op = optimizer.minimize( ... )
  status.assert_consumed()  # Optional sanity checks.
  with tf.compat.v1.Session() as session:
    # Use the Session to restore variables, or initialize them if
    # tf.train.latest_checkpoint returned None.
    status.initialize_or_restore(session)
    for _ in range(num_training_steps):
      session.run(train_op)
    checkpoint.save(file_prefix=checkpoint_prefix)
  ```

  Example usage with eager execution enabled:

  ```python
  import tensorflow as tf
  import os

  tf.compat.v1.enable_eager_execution()

  checkpoint_directory = "/tmp/training_checkpoints"
  checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
  for _ in range(num_training_steps):
    optimizer.minimize( ... )  # Variables will be restored on creation.
  status.assert_consumed()  # Optional sanity checks.
  checkpoint.save(file_prefix=checkpoint_prefix)
  ```

  `Checkpoint.save` and `Checkpoint.restore` write and read object-based
  checkpoints, in contrast to `tf.compat.v1.train.Saver` which writes and reads
  `variable.name` based checkpoints. Object-based checkpointing saves a graph of
  dependencies between Python objects (`Layer`s, `Optimizer`s, `Variable`s,
  etc.) with named edges, and this graph is used to match variables when
  restoring a checkpoint. It can be more robust to changes in the Python
  program, and helps to support restore-on-create for variables when executing
  eagerly. Prefer `tf.train.Checkpoint` over `tf.compat.v1.train.Saver` for new
  code.

  `Checkpoint` objects have dependencies on the objects passed as keyword
  arguments to their constructors, and each dependency is given a name that is
  identical to the name of the keyword argument for which it was created.
  TensorFlow classes like `Layer`s and `Optimizer`s will automatically add
  dependencies on their variables (e.g. "kernel" and "bias" for
  `tf.keras.layers.Dense`). Inheriting from `tf.keras.Model` makes managing
  dependencies easy in user-defined classes, since `Model` hooks into attribute
  assignment. For example:

  ```python
  class Regress(tf.keras.Model):

    def __init__(self):
      super().__init__()
      self.input_transform = tf.keras.layers.Dense(10)
      # ...

    def call(self, inputs):
      x = self.input_transform(inputs)
      # ...
  ```

  This `Model` has a dependency named "input_transform" on its `Dense` layer,
  which in turn depends on its variables. As a result, saving an instance of
  `Regress` using `tf.train.Checkpoint` will also save all the variables created
  by the `Dense` layer.

  When variables are assigned to multiple workers, each worker writes its own
  section of the checkpoint. These sections are then merged/re-indexed to behave
  as a single checkpoint. This avoids copying all variables to one worker, but
  does require that all workers see a common filesystem.

  While `tf.keras.Model.save_weights` and `tf.train.Checkpoint.save` save in the
  same format, note that the root of the resulting checkpoint is the object the
  save method is attached to. This means saving a `tf.keras.Model` using
  `save_weights` and loading into a `tf.train.Checkpoint` with a `Model`
  attached (or vice versa) will not match the `Model`'s variables. See the
  [guide to training
  checkpoints](https://www.tensorflow.org/guide/checkpoint) for
  details. Prefer `tf.train.Checkpoint` over `tf.keras.Model.save_weights` for
  training checkpoints.

  Attributes:
    save_counter: Incremented when `save()` is called. Used to number
      checkpoints.
  """

    def __init__(self, **kwargs):
        """Group objects into a training checkpoint.

    Args:
      **kwargs: Keyword arguments are set as attributes of this object, and are
        saved with the checkpoint. Values must be trackable objects.

    Raises:
      ValueError: If objects in `kwargs` are not trackable.
    """
        super().__init__()
        global _END_TIME_OF_LAST_WRITE
        with _END_TIME_OF_LAST_WRITE_LOCK:
            if _END_TIME_OF_LAST_WRITE is None:
                _END_TIME_OF_LAST_WRITE = time.time()
        for k, v in sorted(kwargs.items(), key=lambda item: item[0]):
            setattr(self, k, v)
            if not isinstance(getattr(self, k), (base.Trackable, def_function.Function)):
                raise ValueError(f'`Checkpoint` was expecting a trackable object (an object derived from `Trackable`), got {v}. If you believe this object should be trackable (i.e. it is part of the TensorFlow Python API and manages state), please open an issue.')
        self._save_counter = None
        self._save_assign_op = None
        self._saver = TrackableSaver(graph_view_lib.ObjectGraphView(self))

    def _maybe_create_save_counter(self):
        """Create a save counter if it does not yet exist."""
        if self._save_counter is None:
            with ops.device('/cpu:0'):
                self._save_counter = data_structures.NoDependency(add_variable(self, name='save_counter', initializer=0, dtype=dtypes.int64, trainable=False))

    def write(self, file_prefix, session=None):
        """Writes a training checkpoint.

    The checkpoint includes variables created by this object and any
    trackable objects it depends on at the time `Checkpoint.write()` is
    called.

    `write` does not number checkpoints, increment `save_counter`, or update the
    metadata used by `tf.train.latest_checkpoint`. It is primarily intended for
    use by higher level checkpoint management utilities. `save` provides a very
    basic implementation of these features.

    Args:
      file_prefix: A prefix to use for the checkpoint filenames
        (/path/to/directory/and_a_prefix).
      session: The session to evaluate variables in. Ignored when executing
        eagerly. If not provided when graph building, the default session is
        used.

    Returns:
      The full path to the checkpoint (i.e. `file_prefix`).
    """
        return self._write(file_prefix, session)

    def _write(self, file_prefix, session=None, write_done_callback=None):
        """Writes a training checkpoint.

    The checkpoint includes variables created by this object and any
    trackable objects it depends on at the time `Checkpoint.write()` is
    called.

    `write` does not number checkpoints, increment `save_counter`, or update the
    metadata used by `tf.train.latest_checkpoint`. It is primarily intended for
    use by higher level checkpoint management utilities. `save` provides a very
    basic implementation of these features.

    Args:
      file_prefix: A prefix to use for the checkpoint filenames
        (/path/to/directory/and_a_prefix).
      session: The session to evaluate variables in. Ignored when executing
        eagerly. If not provided when graph building, the default session is
        used.
      write_done_callback: Optional callback function to be executed once
        the underlying checkpoint saving is finished. Example usage includes
        updating the checkpoint internal state.

    Returns:
      The full path to the checkpoint (i.e. `file_prefix`).
    """
        start_time = time.time()
        output = self._saver.save(file_prefix=file_prefix, session=session)
        end_time = time.time()
        metrics.AddCheckpointWriteDuration(api_label=_CHECKPOINT_V1, microseconds=_get_duration_microseconds(start_time, end_time))
        global _END_TIME_OF_LAST_WRITE
        with _END_TIME_OF_LAST_WRITE_LOCK:
            metrics.AddTrainingTimeSaved(api_label=_CHECKPOINT_V1, microseconds=_get_duration_microseconds(_END_TIME_OF_LAST_WRITE, end_time))
            if checkpoint_context.in_preemption_save_context():
                _preemption_checkpoint_saved_time_usecs.get_cell().increase_by(_get_duration_microseconds(_END_TIME_OF_LAST_WRITE, end_time))
            _END_TIME_OF_LAST_WRITE = end_time
        if tensor_util.is_tf_type(output):
            if context.executing_eagerly():
                output = compat.as_str(output.numpy())
        else:
            output = compat.as_str(output)
        if write_done_callback:
            write_done_callback(output)
        metrics.RecordCheckpointSize(api_label=_CHECKPOINT_V1, filesize=_get_checkpoint_size(output))
        return output

    @property
    def save_counter(self):
        """An integer variable which starts at zero and is incremented on save.

    Used to number checkpoints.

    Returns:
      The save counter variable.
    """
        self._maybe_create_save_counter()
        return self._save_counter

    def save(self, file_prefix, session=None):
        """Saves a training checkpoint and provides basic checkpoint management.

    The saved checkpoint includes variables created by this object and any
    trackable objects it depends on at the time `Checkpoint.save()` is
    called.

    `save` is a basic convenience wrapper around the `write` method,
    sequentially numbering checkpoints using `save_counter` and updating the
    metadata used by `tf.train.latest_checkpoint`. More advanced checkpoint
    management, for example garbage collection and custom numbering, may be
    provided by other utilities which also wrap `write`
    (`tf.train.CheckpointManager` for example).

    Args:
      file_prefix: A prefix to use for the checkpoint filenames
        (/path/to/directory/and_a_prefix). Names are generated based on this
        prefix and `Checkpoint.save_counter`.
      session: The session to evaluate variables in. Ignored when executing
        eagerly. If not provided when graph building, the default session is
        used.

    Returns:
      The full path to the checkpoint.
    """
        graph_building = not context.executing_eagerly()
        if graph_building:
            if ops.inside_function():
                raise NotImplementedError('Calling tf.train.Checkpoint.save() from a function is not supported, as save() modifies saving metadata in ways not supported by TensorFlow Operations. Consider using tf.train.Checkpoint.write(), a lower-level API which does not update metadata. tf.train.latest_checkpoint and related APIs will not see this checkpoint.')
            if session is None:
                session = get_session()
            if self._save_counter is None:
                session.run(self.save_counter.initializer)
        if not graph_building or self._save_assign_op is None:
            with ops.colocate_with(self.save_counter):
                assign_op = self.save_counter.assign_add(1, read_value=True)
            if graph_building:
                self._save_assign_op = data_structures.NoDependency(assign_op)
        if graph_building:
            checkpoint_number = session.run(self._save_assign_op)
        else:
            checkpoint_number = assign_op.numpy()
        file_path = self.write('%s-%d' % (file_prefix, checkpoint_number), session=session)
        checkpoint_management.update_checkpoint_state_internal(save_dir=os.path.dirname(file_prefix), model_checkpoint_path=file_path, all_model_checkpoint_paths=[file_path], save_relative_paths=True)
        return file_path

    def restore(self, save_path):
        """Restore a training checkpoint.

    Restores this `Checkpoint` and any objects it depends on.

    When executing eagerly, either assigns values immediately if variables to
    restore have been created already, or defers restoration until the variables
    are created. Dependencies added after this call will be matched if they have
    a corresponding object in the checkpoint (the restore request will queue in
    any trackable object waiting for the expected dependency to be added).

    When graph building, restoration ops are added to the graph but not run
    immediately.

    ```python
    checkpoint = tf.train.Checkpoint( ... )
    checkpoint.restore(path)
    ```

    To ensure that loading is complete and no more deferred restorations will
    take place, you can use the `assert_consumed()` method of the status object
    returned by `restore`.
    The assert will raise an exception if any Python objects in the dependency
    graph were not found in the checkpoint, or if any checkpointed values do not
    have a matching Python object:

    ```python
    checkpoint = tf.train.Checkpoint( ... )
    checkpoint.restore(path).assert_consumed()
    ```

    When graph building, `assert_consumed()` indicates that all of the restore
    ops that will be created for this checkpoint have been created. They can be
    run via the `run_restore_ops()` method of the status object:

    ```python
    checkpoint.restore(path).assert_consumed().run_restore_ops()
    ```

    If the checkpoint has not been consumed completely, then the list of restore
    ops will grow as more objects are added to the dependency graph.

    To check that all variables in the Python object have restored values from
    checkpoint, use `assert_existing_objects_matched()`. This assertion is
    useful when called after the variables in your graph have been created.

    Name-based `tf.compat.v1.train.Saver` checkpoints can be loaded using this
    method. Names are used to match variables. No restore ops are created/run
    until `run_restore_ops()` or `initialize_or_restore()` are called on the
    returned status object when graph building, but there is restore-on-creation
    when executing eagerly. Re-encode name-based checkpoints using
    `tf.train.Checkpoint.save` as soon as possible.

    Args:
      save_path: The path to the checkpoint, as returned by `save` or
        `tf.train.latest_checkpoint`. If None (as when there is no latest
        checkpoint for `tf.train.latest_checkpoint` to return), returns an
        object which may run initializers for objects in the dependency graph.
        If the checkpoint was written by the name-based
        `tf.compat.v1.train.Saver`, names are used to match variables.

    Returns:
      A load status object, which can be used to make assertions about the
      status of a checkpoint restoration and run initialization/restore ops.

      The returned status object has the following methods:

      * `assert_consumed()`:
          Raises an exception if any variables are unmatched: either
          checkpointed values which don't have a matching Python object or
          Python objects in the dependency graph with no values in the
          checkpoint. This method returns the status object, and so may be
          chained with `initialize_or_restore` or `run_restore_ops`.

      * `assert_existing_objects_matched()`:
          Raises an exception if any existing Python objects in the dependency
          graph are unmatched. Unlike `assert_consumed`, this assertion will
          pass if values in the checkpoint have no corresponding Python
          objects. For example a `tf.keras.Layer` object which has not yet been
          built, and so has not created any variables, will pass this assertion
          but will fail `assert_consumed`. Useful when loading part of a larger
          checkpoint into a new Python program, e.g. a training checkpoint with
          a `tf.compat.v1.train.Optimizer` was saved but only the state required
          for inference is being loaded. This method returns the status object,
          and so may be chained with `initialize_or_restore` or
          `run_restore_ops`.

      * `assert_nontrivial_match()`: Asserts that something aside from the root
          object was matched. This is a very weak assertion, but is useful for
          sanity checking in library code where objects may exist in the
          checkpoint which haven't been created in Python and some Python
          objects may not have a checkpointed value.

      * `expect_partial()`: Silence warnings about incomplete checkpoint
          restores. Warnings are otherwise printed for unused parts of the
          checkpoint file or object when the `Checkpoint` object is deleted
          (often at program shutdown).

      * `initialize_or_restore(session=None)`:
          When graph building, runs variable initializers if `save_path` is
          `None`, but otherwise runs restore operations. If no `session` is
          explicitly specified, the default session is used. No effect when
          executing eagerly (variables are initialized or restored eagerly).

      * `run_restore_ops(session=None)`:
          When graph building, runs restore operations. If no `session` is
          explicitly specified, the default session is used. No effect when
          executing eagerly (restore operations are run eagerly). May only be
          called when `save_path` is not `None`.
    """
        start_time = time.time()
        status = self._saver.restore(save_path=save_path)
        self._maybe_create_save_counter()
        if isinstance(status, NameBasedSaverStatus):
            status.add_to_optionally_restored(self.save_counter)
        metrics.AddCheckpointReadDuration(api_label=_CHECKPOINT_V1, microseconds=_get_duration_microseconds(start_time, time.time()))
        return status