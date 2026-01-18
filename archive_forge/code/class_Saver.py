import collections
import glob
import os.path
import threading
import time
import numpy as np
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import training_util
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.Saver'])
class Saver:
    """Saves and restores variables.

  @compatibility(TF2)
  `tf.compat.v1.train.Saver` is not supported for saving and restoring
  checkpoints in TF2. Please switch to `tf.train.Checkpoint` or
  `tf.keras.Model.save_weights`, which perform a more robust [object-based
  saving](https://www.tensorflow.org/guide/checkpoint#loading_mechanics).

  ### How to Rewrite Checkpoints

  Please rewrite your checkpoints immediately using the object-based checkpoint
  APIs.

  You can load a name-based checkpoint written by `tf.compat.v1.train.Saver`
  using `tf.train.Checkpoint.restore` or `tf.keras.Model.load_weights`. However,
  you may have to change the names of the variables in your model to match the
  variable names in the name-based checkpoint, which can be viewed with
  `tf.train.list_variables(path)`.

  Another option is to create an `assignment_map` that maps the name of the
  variables in the name-based checkpoint to the variables in your model, eg:
  ```
  {
      'sequential/dense/bias': model.variables[0],
      'sequential/dense/kernel': model.variables[1]
  }
  ```
  and use `tf.compat.v1.train.init_from_checkpoint(path, assignment_map)` to
  restore the name-based checkpoint.

  After restoring, re-encode your checkpoint
  using `tf.train.Checkpoint.save` or `tf.keras.Model.save_weights`.

  See the [Checkpoint compatibility](
  https://www.tensorflow.org/guide/migrate#checkpoint_compatibility)
  section of the migration guide for more details.


  ### Checkpoint Management in TF2

  Use `tf.train.CheckpointManager` to manage checkpoints in TF2.
  `tf.train.CheckpointManager` offers equivalent `keep_checkpoint_every_n_hours`
  and `max_to_keep` parameters.

  To recover the latest checkpoint,

  ```
  checkpoint = tf.train.Checkpoint(model)
  manager = tf.train.CheckpointManager(checkpoint)
  status = checkpoint.restore(manager.latest_checkpoint)
  ```

  `tf.train.CheckpointManager` also writes a [`CheckpointState` proto]
  (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/checkpoint_state.proto)
  which contains the timestamp when each checkpoint was created.

  ### Writing `MetaGraphDef`s in TF2

  To replace, `tf.compat.v1.train.Saver.save(write_meta_graph=True)`, use
  `tf.saved_model.save` to write the `MetaGraphDef` (which is contained in
  `saved_model.pb`).

  @end_compatibility

  See [Variables](https://tensorflow.org/guide/variables)
  for an overview of variables, saving and restoring.

  The `Saver` class adds ops to save and restore variables to and from
  *checkpoints*.  It also provides convenience methods to run these ops.

  Checkpoints are binary files in a proprietary format which map variable names
  to tensor values.  The best way to examine the contents of a checkpoint is to
  load it using a `Saver`.

  Savers can automatically number checkpoint filenames with a provided counter.
  This lets you keep multiple checkpoints at different steps while training a
  model.  For example you can number the checkpoint filenames with the training
  step number.  To avoid filling up disks, savers manage checkpoint files
  automatically. For example, they can keep only the N most recent files, or
  one checkpoint for every N hours of training.

  You number checkpoint filenames by passing a value to the optional
  `global_step` argument to `save()`:

  ```python
  saver.save(sess, 'my-model', global_step=0) ==> filename: 'my-model-0'
  ...
  saver.save(sess, 'my-model', global_step=1000) ==> filename: 'my-model-1000'
  ```

  Additionally, optional arguments to the `Saver()` constructor let you control
  the proliferation of checkpoint files on disk:

  * `max_to_keep` indicates the maximum number of recent checkpoint files to
    keep.  As new files are created, older files are deleted.   If None or 0,
    no checkpoints are deleted from the filesystem but only the last one is
    kept in the `checkpoint` file.  Defaults to 5 (that is, the 5 most recent
    checkpoint files are kept.)

  * `keep_checkpoint_every_n_hours`: In addition to keeping the most recent
    `max_to_keep` checkpoint files, you might want to keep one checkpoint file
    for every N hours of training.  This can be useful if you want to later
    analyze how a model progressed during a long training session.  For
    example, passing `keep_checkpoint_every_n_hours=2` ensures that you keep
    one checkpoint file for every 2 hours of training.  The default value of
    10,000 hours effectively disables the feature.

  Note that you still have to call the `save()` method to save the model.
  Passing these arguments to the constructor will not save variables
  automatically for you.

  A training program that saves regularly looks like:

  ```python
  ...
  # Create a saver.
  saver = tf.compat.v1.train.Saver(...variables...)
  # Launch the graph and train, saving the model every 1,000 steps.
  sess = tf.compat.v1.Session()
  for step in range(1000000):
      sess.run(..training_op..)
      if step % 1000 == 0:
          # Append the step number to the checkpoint name:
          saver.save(sess, 'my-model', global_step=step)
  ```

  In addition to checkpoint files, savers keep a protocol buffer on disk with
  the list of recent checkpoints. This is used to manage numbered checkpoint
  files and by `latest_checkpoint()`, which makes it easy to discover the path
  to the most recent checkpoint. That protocol buffer is stored in a file named
  'checkpoint' next to the checkpoint files.

  If you create several savers, you can specify a different filename for the
  protocol buffer file in the call to `save()`.
  """

    def __init__(self, var_list=None, reshape=False, sharded=False, max_to_keep=5, keep_checkpoint_every_n_hours=10000.0, name=None, restore_sequentially=False, saver_def=None, builder=None, defer_build=False, allow_empty=False, write_version=saver_pb2.SaverDef.V2, pad_step_number=False, save_relative_paths=False, filename=None):
        """Creates a `Saver`.

    The constructor adds ops to save and restore variables.

    `var_list` specifies the variables that will be saved and restored. It can
    be passed as a `dict` or a list:

    * A `dict` of names to variables: The keys are the names that will be
      used to save or restore the variables in the checkpoint files.
    * A list of variables: The variables will be keyed with their op name in
      the checkpoint files.

    For example:

    ```python
    v1 = tf.Variable(..., name='v1')
    v2 = tf.Variable(..., name='v2')

    # Pass the variables as a dict:
    saver = tf.compat.v1.train.Saver({'v1': v1, 'v2': v2})

    # Or pass them as a list.
    saver = tf.compat.v1.train.Saver([v1, v2])
    # Passing a list is equivalent to passing a dict with the variable op names
    # as keys:
    saver = tf.compat.v1.train.Saver({v.op.name: v for v in [v1, v2]})
    ```

    Note: the newer `AutoTrackable` API is not supported by `Saver`. In this
    case, the `tf.train.Checkpoint` class should be used.

    The optional `reshape` argument, if `True`, allows restoring a variable from
    a save file where the variable had a different shape, but the same number
    of elements and type.  This is useful if you have reshaped a variable and
    want to reload it from an older checkpoint.

    The optional `sharded` argument, if `True`, instructs the saver to shard
    checkpoints per device.

    Args:
      var_list: A list of `Variable`/`SaveableObject`, or a dictionary mapping
        names to `SaveableObject`s. If `None`, defaults to the list of all
        saveable objects.
      reshape: If `True`, allows restoring parameters from a checkpoint where
        the variables have a different shape.
      sharded: If `True`, shard the checkpoints, one per device.
      max_to_keep: Maximum number of recent checkpoints to keep. Defaults to 5.
      keep_checkpoint_every_n_hours: How often to keep checkpoints. Defaults to
        10,000 hours.
      name: String.  Optional name to use as a prefix when adding operations.
      restore_sequentially: A `Bool`, which if true, causes restore of different
        variables to happen sequentially within each device.  This can lower
        memory usage when restoring very large models.
      saver_def: Optional `SaverDef` proto to use instead of running the
        builder. This is only useful for specialty code that wants to recreate a
        `Saver` object for a previously built `Graph` that had a `Saver`. The
        `saver_def` proto should be the one returned by the `as_saver_def()`
        call of the `Saver` that was created for that `Graph`.
      builder: Optional `SaverBuilder` to use if a `saver_def` was not provided.
        Defaults to `BulkSaverBuilder()`.
      defer_build: If `True`, defer adding the save and restore ops to the
        `build()` call. In that case `build()` should be called before
        finalizing the graph or using the saver.
      allow_empty: If `False` (default) raise an error if there are no variables
        in the graph. Otherwise, construct the saver anyway and make it a no-op.
      write_version: controls what format to use when saving checkpoints.  It
        also affects certain filepath matching logic.  The V2 format is the
        recommended choice: it is much more optimized than V1 in terms of memory
        required and latency incurred during restore.  Regardless of this flag,
        the Saver is able to restore from both V2 and V1 checkpoints.
      pad_step_number: if True, pads the global step number in the checkpoint
        filepaths to some fixed width (8 by default).  This is turned off by
        default.
      save_relative_paths: If `True`, will write relative paths to the
        checkpoint state file. This is needed if the user wants to copy the
        checkpoint directory and reload from the copied directory.
      filename: If known at graph construction time, filename used for variable
        loading/saving.

    Raises:
      TypeError: If `var_list` is invalid.
      ValueError: If any of the keys or values in `var_list` are not unique.
      RuntimeError: If eager execution is enabled and`var_list` does not specify
        a list of variables to save.

    @compatibility(eager)
    When eager execution is enabled, `var_list` must specify a `list` or `dict`
    of variables to save. Otherwise, a `RuntimeError` will be raised.

    Although Saver works in some cases when executing eagerly, it is
    fragile. Please switch to `tf.train.Checkpoint` or
    `tf.keras.Model.save_weights`, which perform a more robust object-based
    saving. These APIs will load checkpoints written by `Saver`.
    @end_compatibility
    """
        global _END_TIME_OF_LAST_WRITE
        with _END_TIME_OF_LAST_WRITE_LOCK:
            if _END_TIME_OF_LAST_WRITE is None:
                _END_TIME_OF_LAST_WRITE = time.time()
        if defer_build and var_list:
            raise ValueError('If `var_list` is provided then build cannot be deferred. Either set defer_build=False or var_list=None.')
        if context.executing_eagerly():
            logging.warning('Saver is deprecated, please switch to tf.train.Checkpoint or tf.keras.Model.save_weights for training checkpoints. When executing eagerly variables do not necessarily have unique names, and so the variable.name-based lookups Saver performs are error-prone.')
            if var_list is None:
                raise RuntimeError('When eager execution is enabled, `var_list` must specify a list or dict of variables to save')
        self._var_list = var_list
        self._reshape = reshape
        self._sharded = sharded
        self._max_to_keep = max_to_keep
        self._keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
        self._name = name
        self._restore_sequentially = restore_sequentially
        self.saver_def = saver_def
        self._builder = builder
        self._is_built = False
        self._allow_empty = allow_empty
        self._is_empty = None
        self._write_version = write_version
        self._pad_step_number = pad_step_number
        self._filename = filename
        self._last_checkpoints = []
        self._checkpoints_to_be_deleted = []
        if context.executing_eagerly():
            self._next_checkpoint_time = time.time() + self._keep_checkpoint_every_n_hours * 3600
        elif not defer_build:
            self.build()
        if self.saver_def:
            self._check_saver_def()
            self._write_version = self.saver_def.version
        self._save_relative_paths = save_relative_paths
        self._object_restore_saver = None

    def build(self):
        if context.executing_eagerly():
            raise RuntimeError('Use save/restore instead of build in eager mode.')
        self._build(self._filename, build_save=True, build_restore=True)

    def _build_eager(self, checkpoint_path, build_save, build_restore):
        self._build(checkpoint_path, build_save=build_save, build_restore=build_restore)

    def _build(self, checkpoint_path, build_save, build_restore):
        """Builds saver_def."""
        if not context.executing_eagerly():
            if self._is_built:
                return
            self._is_built = True
        if not self.saver_def or context.executing_eagerly():
            if self._builder is None:
                self._builder = BulkSaverBuilder(self._write_version)
            if self._var_list is None:
                self._var_list = variables._all_saveable_objects()
            if not self._var_list:
                if self._allow_empty:
                    self._is_empty = True
                    return
                else:
                    raise ValueError('No variables to save')
            self._is_empty = False
            self.saver_def = self._builder._build_internal(self._var_list, reshape=self._reshape, sharded=self._sharded, max_to_keep=self._max_to_keep, keep_checkpoint_every_n_hours=self._keep_checkpoint_every_n_hours, name=self._name, restore_sequentially=self._restore_sequentially, filename=checkpoint_path, build_save=build_save, build_restore=build_restore)
        elif self.saver_def and self._name:
            self.saver_def.filename_tensor_name = ops.prepend_name_scope(self.saver_def.filename_tensor_name, self._name)
            self.saver_def.save_tensor_name = ops.prepend_name_scope(self.saver_def.save_tensor_name, self._name)
            self.saver_def.restore_op_name = ops.prepend_name_scope(self.saver_def.restore_op_name, self._name)
        self._check_saver_def()
        if not context.executing_eagerly():
            self._next_checkpoint_time = time.time() + self.saver_def.keep_checkpoint_every_n_hours * 3600

    def _check_saver_def(self):
        if not isinstance(self.saver_def, saver_pb2.SaverDef):
            raise ValueError('saver_def must be a saver_pb2.SaverDef: %s' % self.saver_def)
        if not context.executing_eagerly():
            if not self.saver_def.save_tensor_name:
                raise ValueError('saver_def must specify the save_tensor_name: %s' % str(self.saver_def))
            if not self.saver_def.restore_op_name:
                raise ValueError('saver_def must specify the restore_op_name: %s' % str(self.saver_def))

    def _CheckpointFilename(self, p):
        """Returns the checkpoint filename given a `(filename, time)` pair.

    Args:
      p: (filename, time) pair.

    Returns:
      Checkpoint file name.
    """
        name, _ = p
        return name

    def _RecordLastCheckpoint(self, latest_save_path):
        """Manages the list of the latest checkpoints."""
        if not self.saver_def.max_to_keep:
            return
        for p in self._last_checkpoints:
            if latest_save_path == self._CheckpointFilename(p):
                self._last_checkpoints.remove(p)
        self._last_checkpoints.append((latest_save_path, time.time()))
        if len(self._last_checkpoints) > self.saver_def.max_to_keep:
            self._checkpoints_to_be_deleted.append(self._last_checkpoints.pop(0))

    def _MaybeDeleteOldCheckpoints(self, meta_graph_suffix='meta'):
        """Deletes old checkpoints if necessary.

    `self._checkpoints_to_be_deleted` is going to contain checkpoints that are
    over `max_to_keep`.  They are going to be deleted.  If
    `keep_checkpoint_every_n_hours` was specified, keep an additional checkpoint
    every `N` hours. For example, if `N` is 0.5, an additional checkpoint is
    kept for every 0.5 hours of training; if `N` is 10, an additional
    checkpoint is kept for every 10 hours of training.

    Args:
      meta_graph_suffix: Suffix for `MetaGraphDef` file. Defaults to 'meta'.
    """
        if self._checkpoints_to_be_deleted:
            p = self._checkpoints_to_be_deleted.pop(0)
            should_keep = p[1] > self._next_checkpoint_time
            if should_keep:
                self._next_checkpoint_time += self.saver_def.keep_checkpoint_every_n_hours * 3600
                return
            try:
                checkpoint_management.remove_checkpoint(self._CheckpointFilename(p), self.saver_def.version, meta_graph_suffix)
            except Exception as e:
                logging.warning('Ignoring: %s', str(e))

    def as_saver_def(self):
        """Generates a `SaverDef` representation of this saver.

    Returns:
      A `SaverDef` proto.
    """
        return self.saver_def

    def to_proto(self, export_scope=None):
        """Converts this `Saver` to a `SaverDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Returns:
      A `SaverDef` protocol buffer.
    """
        if export_scope is None:
            return self.saver_def
        if not (self.saver_def.filename_tensor_name.startswith(export_scope) and self.saver_def.save_tensor_name.startswith(export_scope) and self.saver_def.restore_op_name.startswith(export_scope)):
            return None
        saver_def = saver_pb2.SaverDef()
        saver_def.CopyFrom(self.saver_def)
        saver_def.filename_tensor_name = ops.strip_name_scope(saver_def.filename_tensor_name, export_scope)
        saver_def.save_tensor_name = ops.strip_name_scope(saver_def.save_tensor_name, export_scope)
        saver_def.restore_op_name = ops.strip_name_scope(saver_def.restore_op_name, export_scope)
        return saver_def

    @staticmethod
    def from_proto(saver_def, import_scope=None):
        """Returns a `Saver` object created from `saver_def`.

    Args:
      saver_def: a `SaverDef` protocol buffer.
      import_scope: Optional `string`. Name scope to use.

    Returns:
      A `Saver` built from saver_def.
    """
        return Saver(saver_def=saver_def, name=import_scope)

    @property
    def last_checkpoints(self):
        """List of not-yet-deleted checkpoint filenames.

    You can pass any of the returned values to `restore()`.

    Returns:
      A list of checkpoint filenames, sorted from oldest to newest.
    """
        return list((self._CheckpointFilename(p) for p in self._last_checkpoints))

    def set_last_checkpoints(self, last_checkpoints):
        """DEPRECATED: Use set_last_checkpoints_with_time.

    Sets the list of old checkpoint filenames.

    Args:
      last_checkpoints: A list of checkpoint filenames.

    Raises:
      AssertionError: If last_checkpoints is not a list.
    """
        assert isinstance(last_checkpoints, list)
        self._last_checkpoints = [(s, np.inf) for s in last_checkpoints]

    def set_last_checkpoints_with_time(self, last_checkpoints_with_time):
        """Sets the list of old checkpoint filenames and timestamps.

    Args:
      last_checkpoints_with_time: A list of tuples of checkpoint filenames and
        timestamps.

    Raises:
      AssertionError: If last_checkpoints_with_time is not a list.
    """
        assert isinstance(last_checkpoints_with_time, list)
        self._last_checkpoints = last_checkpoints_with_time

    def recover_last_checkpoints(self, checkpoint_paths):
        """Recovers the internal saver state after a crash.

    This method is useful for recovering the "self._last_checkpoints" state.

    Globs for the checkpoints pointed to by `checkpoint_paths`.  If the files
    exist, use their mtime as the checkpoint timestamp.

    Args:
      checkpoint_paths: a list of checkpoint paths.
    """
        checkpoints_with_mtimes = []
        for checkpoint_path in checkpoint_paths:
            try:
                mtime = checkpoint_management.get_checkpoint_mtimes([checkpoint_path])
            except errors.NotFoundError:
                continue
            if mtime:
                checkpoints_with_mtimes.append((checkpoint_path, mtime[0]))
        self.set_last_checkpoints_with_time(checkpoints_with_mtimes)

    def save(self, sess, save_path, global_step=None, latest_filename=None, meta_graph_suffix='meta', write_meta_graph=True, write_state=True, strip_default_attrs=False, save_debug_info=False):
        """Saves variables.

    This method runs the ops added by the constructor for saving variables.
    It requires a session in which the graph was launched.  The variables to
    save must also have been initialized.

    The method returns the path prefix of the newly created checkpoint files.
    This string can be passed directly to a call to `restore()`.

    Args:
      sess: A Session to use to save the variables.
      save_path: String.  Prefix of filenames created for the checkpoint.
      global_step: If provided the global step number is appended to `save_path`
        to create the checkpoint filenames. The optional argument can be a
        `Tensor`, a `Tensor` name or an integer.
      latest_filename: Optional name for the protocol buffer file that will
        contains the list of most recent checkpoints.  That file, kept in the
        same directory as the checkpoint files, is automatically managed by the
        saver to keep track of recent checkpoints.  Defaults to 'checkpoint'.
      meta_graph_suffix: Suffix for `MetaGraphDef` file. Defaults to 'meta'.
      write_meta_graph: `Boolean` indicating whether or not to write the meta
        graph file.
      write_state: `Boolean` indicating whether or not to write the
        `CheckpointStateProto`.
      strip_default_attrs: Boolean. If `True`, default-valued attributes will be
        removed from the NodeDefs. For a detailed guide, see [Stripping
        Default-Valued
        Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).
      save_debug_info: If `True`, save the GraphDebugInfo to a separate file,
        which in the same directory of save_path and with `_debug` added before
        the file extension. This is only enabled when `write_meta_graph` is
        `True`

    Returns:
      A string: path prefix used for the checkpoint files.  If the saver is
        sharded, this string ends with: '-?????-of-nnnnn' where 'nnnnn'
        is the number of shards created.
      If the saver is empty, returns None.

    Raises:
      TypeError: If `sess` is not a `Session`.
      ValueError: If `latest_filename` contains path components, or if it
        collides with `save_path`.
      RuntimeError: If save and restore ops weren't built.
    """
        start_time = time.time()
        if not self._is_built and (not context.executing_eagerly()):
            raise RuntimeError('`build()` should be called before save if defer_build==True')
        if latest_filename is None:
            latest_filename = 'checkpoint'
        if self._write_version != saver_pb2.SaverDef.V2:
            logging.warning('*******************************************************')
            logging.warning("TensorFlow's V1 checkpoint format has been deprecated.")
            logging.warning('Consider switching to the more efficient V2 format:')
            logging.warning('   `tf.train.Saver(write_version=tf.train.SaverDef.V2)`')
            logging.warning('now on by default.')
            logging.warning('*******************************************************')
        if os.path.split(latest_filename)[0]:
            raise ValueError("'latest_filename' must not contain path components")
        save_path = compat.as_str(save_path)
        if global_step is not None:
            if not isinstance(global_step, compat.integral_types):
                global_step = training_util.global_step(sess, global_step)
            checkpoint_file = '%s-%d' % (save_path, global_step)
            if self._pad_step_number:
                checkpoint_file = '%s-%s' % (save_path, '{:08d}'.format(global_step))
        else:
            checkpoint_file = save_path
            if os.path.basename(save_path) == latest_filename and (not self._sharded):
                raise ValueError("'latest_filename' collides with 'save_path': '%s' and '%s'" % (latest_filename, save_path))
        if not context.executing_eagerly() and (not isinstance(sess, session.SessionInterface)):
            raise TypeError("'sess' must be a Session; %s" % sess)
        save_path_parent = os.path.dirname(save_path)
        if not self._is_empty:
            try:
                if context.executing_eagerly():
                    self._build_eager(checkpoint_file, build_save=True, build_restore=False)
                    model_checkpoint_path = self.saver_def.save_tensor_name
                else:
                    model_checkpoint_path = sess.run(self.saver_def.save_tensor_name, {self.saver_def.filename_tensor_name: checkpoint_file})
                model_checkpoint_path = compat.as_str(model_checkpoint_path)
                if write_state:
                    self._RecordLastCheckpoint(model_checkpoint_path)
                    checkpoint_management.update_checkpoint_state_internal(save_dir=save_path_parent, model_checkpoint_path=model_checkpoint_path, all_model_checkpoint_paths=self.last_checkpoints, latest_filename=latest_filename, save_relative_paths=self._save_relative_paths)
                    self._MaybeDeleteOldCheckpoints(meta_graph_suffix=meta_graph_suffix)
            except (errors.FailedPreconditionError, errors.NotFoundError) as exc:
                if not gfile.IsDirectory(save_path_parent):
                    exc = ValueError("Parent directory of {} doesn't exist, can't save.".format(save_path))
                raise exc
        end_time = time.time()
        metrics.AddCheckpointWriteDuration(api_label=_SAVER_LABEL, microseconds=_get_duration_microseconds(start_time, end_time))
        global _END_TIME_OF_LAST_WRITE
        with _END_TIME_OF_LAST_WRITE_LOCK:
            metrics.AddTrainingTimeSaved(api_label=_SAVER_LABEL, microseconds=_get_duration_microseconds(_END_TIME_OF_LAST_WRITE, end_time))
            _END_TIME_OF_LAST_WRITE = end_time
        if write_meta_graph:
            meta_graph_filename = checkpoint_management.meta_graph_filename(checkpoint_file, meta_graph_suffix=meta_graph_suffix)
            if not context.executing_eagerly():
                with sess.graph.as_default():
                    self.export_meta_graph(meta_graph_filename, strip_default_attrs=strip_default_attrs, save_debug_info=save_debug_info)
        if self._is_empty:
            return None
        else:
            metrics.RecordCheckpointSize(api_label=_SAVER_LABEL, filesize=_get_checkpoint_size(model_checkpoint_path))
            return model_checkpoint_path

    def export_meta_graph(self, filename=None, collection_list=None, as_text=False, export_scope=None, clear_devices=False, clear_extraneous_savers=False, strip_default_attrs=False, save_debug_info=False):
        """Writes `MetaGraphDef` to save_path/filename.

    Args:
      filename: Optional meta_graph filename including the path.
      collection_list: List of string keys to collect.
      as_text: If `True`, writes the meta_graph as an ASCII proto.
      export_scope: Optional `string`. Name scope to remove.
      clear_devices: Whether or not to clear the device field for an `Operation`
        or `Tensor` during export.
      clear_extraneous_savers: Remove any Saver-related information from the
        graph (both Save/Restore ops and SaverDefs) that are not associated with
        this Saver.
      strip_default_attrs: Boolean. If `True`, default-valued attributes will be
        removed from the NodeDefs. For a detailed guide, see [Stripping
        Default-Valued
        Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).
      save_debug_info: If `True`, save the GraphDebugInfo to a separate file,
        which in the same directory of filename and with `_debug` added before
        the file extension.

    Returns:
      A `MetaGraphDef` proto.
    """
        return export_meta_graph(filename=filename, graph_def=ops.get_default_graph().as_graph_def(add_shapes=True), saver_def=self.saver_def, collection_list=collection_list, as_text=as_text, export_scope=export_scope, clear_devices=clear_devices, clear_extraneous_savers=clear_extraneous_savers, strip_default_attrs=strip_default_attrs, save_debug_info=save_debug_info)

    def restore(self, sess, save_path):
        """Restores previously saved variables.

    This method runs the ops added by the constructor for restoring variables.
    It requires a session in which the graph was launched.  The variables to
    restore do not have to have been initialized, as restoring is itself a way
    to initialize variables.

    The `save_path` argument is typically a value previously returned from a
    `save()` call, or a call to `latest_checkpoint()`.

    Args:
      sess: A `Session` to use to restore the parameters. None in eager mode.
      save_path: Path where parameters were previously saved.

    Raises:
      ValueError: If save_path is None or not a valid checkpoint.
    """
        start_time = time.time()
        if self._is_empty:
            return
        if save_path is None:
            raise ValueError("Can't load save_path when it is None.")
        checkpoint_prefix = compat.as_text(save_path)
        if not checkpoint_management.checkpoint_exists_internal(checkpoint_prefix):
            raise ValueError('The passed save_path is not a valid checkpoint: ' + checkpoint_prefix)
        logging.info('Restoring parameters from %s', checkpoint_prefix)
        try:
            if context.executing_eagerly():
                self._build_eager(save_path, build_save=False, build_restore=True)
            else:
                sess.run(self.saver_def.restore_op_name, {self.saver_def.filename_tensor_name: save_path})
        except errors.NotFoundError as err:
            try:
                names_to_keys = object_graph_key_mapping(save_path)
            except errors.NotFoundError:
                raise _wrap_restore_error_with_msg(err, 'a Variable name or other graph key that is missing')
            logging.warning('Restoring an object-based checkpoint using a name-based saver. This may be somewhat fragile, and will re-build the Saver. Instead, consider loading object-based checkpoints using tf.train.Checkpoint().')
            self._object_restore_saver = saver_from_object_based_checkpoint(checkpoint_path=save_path, var_list=self._var_list, builder=self._builder, names_to_keys=names_to_keys, cached_saver=self._object_restore_saver)
            self._object_restore_saver.restore(sess=sess, save_path=save_path)
        except errors.InvalidArgumentError as err:
            raise _wrap_restore_error_with_msg(err, 'a mismatch between the current graph and the graph')
        metrics.AddCheckpointReadDuration(api_label=_SAVER_LABEL, microseconds=_get_duration_microseconds(start_time, time.time()))

    @staticmethod
    def _add_collection_def(meta_graph_def, key, export_scope=None):
        """Adds a collection to MetaGraphDef protocol buffer.

    Args:
      meta_graph_def: MetaGraphDef protocol buffer.
      key: One of the GraphKeys or user-defined string.
      export_scope: Optional `string`. Name scope to remove.
    """
        meta_graph.add_collection_def(meta_graph_def, key, export_scope=export_scope)