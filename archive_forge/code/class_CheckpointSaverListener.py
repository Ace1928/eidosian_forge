import os
import time
import numpy as np
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.client import timeline
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.CheckpointSaverListener'])
class CheckpointSaverListener:
    """Interface for listeners that take action before or after checkpoint save.

  `CheckpointSaverListener` triggers only in steps when `CheckpointSaverHook` is
  triggered, and provides callbacks at the following points:
   - before using the session
   - before each call to `Saver.save()`
   - after each call to `Saver.save()`
   - at the end of session

  To use a listener, implement a class and pass the listener to a
  `CheckpointSaverHook`, as in this example:

  ```python
  class ExampleCheckpointSaverListener(CheckpointSaverListener):
    def begin(self):
      # You can add ops to the graph here.
      print('Starting the session.')
      self.your_tensor = ...

    def before_save(self, session, global_step_value):
      print('About to write a checkpoint')

    def after_save(self, session, global_step_value):
      print('Done writing checkpoint.')
      if decided_to_stop_training():
        return True

    def end(self, session, global_step_value):
      print('Done with the session.')

  ...
  listener = ExampleCheckpointSaverListener()
  saver_hook = tf.estimator.CheckpointSaverHook(
      checkpoint_dir, listeners=[listener])
  with
  tf.compat.v1.train.MonitoredTrainingSession(chief_only_hooks=[saver_hook]):
    ...
  ```

  A `CheckpointSaverListener` may simply take some action after every
  checkpoint save. It is also possible for the listener to use its own schedule
  to act less frequently, e.g. based on global_step_value. In this case,
  implementors should implement the `end()` method to handle actions related to
  the last checkpoint save. But the listener should not act twice if
  `after_save()` already handled this last checkpoint save.

  A `CheckpointSaverListener` can request training to be stopped, by returning
  True in `after_save`. Please note that, in replicated distributed training
  setting, only `chief` should use this behavior. Otherwise each worker will do
  their own evaluation, which may be wasteful of resources.
  """

    def begin(self):
        pass

    def before_save(self, session, global_step_value):
        pass

    def after_save(self, session, global_step_value):
        pass

    def end(self, session, global_step_value):
        pass