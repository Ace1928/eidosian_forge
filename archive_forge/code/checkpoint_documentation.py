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
Restores a training checkpoint.

    Restores this `Checkpoint` and any objects it depends on.

    This method is intended to be used to load checkpoints created by `save()`.
    For checkpoints created by `write()` use the `read()` method which does not
    expect the `save_counter` variable added by `save()`.

    `restore()` either assigns values immediately if variables to restore have
    been created already, or defers restoration until the variables are
    created. Dependencies added after this call will be matched if they have a
    corresponding object in the checkpoint (the restore request will queue in
    any trackable object waiting for the expected dependency to be added).

    ```python
    checkpoint = tf.train.Checkpoint( ... )
    checkpoint.restore(path)

    # You can additionally pass options to restore():
    options = tf.CheckpointOptions(experimental_io_device="/job:localhost")
    checkpoint.restore(path, options=options)
    ```

    To ensure that loading is complete and no more deferred restorations will
    take place, use the `assert_consumed()` method of the status object returned
    by `restore()`:

    ```python
    checkpoint.restore(path, options=options).assert_consumed()
    ```

    The assert will raise an error if any Python objects in the dependency graph
    were not found in the checkpoint, or if any checkpointed values do not have
    a matching Python object.

    Name-based `tf.compat.v1.train.Saver` checkpoints from TensorFlow 1.x can be
    loaded using this method. Names are used to match variables. Re-encode
    name-based checkpoints using `tf.train.Checkpoint.save` as soon as possible.

    **Loading from SavedModel checkpoints**

    To load values from a SavedModel, just pass the SavedModel directory
    to checkpoint.restore:

    ```python
    model = tf.keras.Model(...)
    tf.saved_model.save(model, path)  # or model.save(path, save_format='tf')

    checkpoint = tf.train.Checkpoint(model)
    checkpoint.restore(path).expect_partial()
    ```

    This example calls `expect_partial()` on the loaded status, since
    SavedModels saved from Keras often generates extra keys in the checkpoint.
    Otherwise, the program prints a lot of warnings about unused keys at exit
    time.

    Args:
      save_path: The path to the checkpoint, as returned by `save` or
        `tf.train.latest_checkpoint`. If the checkpoint was written by the
        name-based `tf.compat.v1.train.Saver`, names are used to match
        variables. This path may also be a SavedModel directory.
      options: Optional `tf.train.CheckpointOptions` object.

    Returns:
      A load status object, which can be used to make assertions about the
      status of a checkpoint restoration.

      The returned status object has the following methods:

      * `assert_consumed()`:
          Raises an exception if any variables are unmatched: either
          checkpointed values which don't have a matching Python object or
          Python objects in the dependency graph with no values in the
          checkpoint. This method returns the status object, and so may be
          chained with other assertions.

      * `assert_existing_objects_matched()`:
          Raises an exception if any existing Python objects in the dependency
          graph are unmatched. Unlike `assert_consumed`, this assertion will
          pass if values in the checkpoint have no corresponding Python
          objects. For example a `tf.keras.Layer` object which has not yet been
          built, and so has not created any variables, will pass this assertion
          but fail `assert_consumed`. Useful when loading part of a larger
          checkpoint into a new Python program, e.g. a training checkpoint with
          a `tf.compat.v1.train.Optimizer` was saved but only the state required
          for
          inference is being loaded. This method returns the status object, and
          so may be chained with other assertions.

      * `assert_nontrivial_match()`: Asserts that something aside from the root
          object was matched. This is a very weak assertion, but is useful for
          sanity checking in library code where objects may exist in the
          checkpoint which haven't been created in Python and some Python
          objects may not have a checkpointed value.

      * `expect_partial()`: Silence warnings about incomplete checkpoint
          restores. Warnings are otherwise printed for unused parts of the
          checkpoint file or object when the `Checkpoint` object is deleted
          (often at program shutdown).

    Raises:
      NotFoundError: if the a checkpoint or SavedModel cannot be found at
        `save_path`.
    