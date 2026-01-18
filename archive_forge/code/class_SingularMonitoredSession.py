import abc
import os
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.distribute import distribute_coordinator_context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner
from tensorflow.python.training import saver as training_saver
from tensorflow.python.training import session_manager as sm
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import function_utils
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.SingularMonitoredSession'])
class SingularMonitoredSession(_MonitoredSession):
    """Session-like object that handles initialization, restoring, and hooks.

  Please note that this utility is not recommended for distributed settings.
  For distributed settings, please use `tf.compat.v1.train.MonitoredSession`.
  The
  differences between `MonitoredSession` and `SingularMonitoredSession` are:

  * `MonitoredSession` handles `AbortedError` and `UnavailableError` for
    distributed settings, but `SingularMonitoredSession` does not.
  * `MonitoredSession` can be created in `chief` or `worker` modes.
    `SingularMonitoredSession` is always created as `chief`.
  * You can access the raw `tf.compat.v1.Session` object used by
    `SingularMonitoredSession`, whereas in MonitoredSession the raw session is
    private. This can be used:
      - To `run` without hooks.
      - To save and restore.
  * All other functionality is identical.

  Example usage:
  ```python
  saver_hook = CheckpointSaverHook(...)
  summary_hook = SummarySaverHook(...)
  with SingularMonitoredSession(hooks=[saver_hook, summary_hook]) as sess:
    while not sess.should_stop():
      sess.run(train_op)
  ```

  Initialization: At creation time the hooked session does following things
  in given order:

  * calls `hook.begin()` for each given hook
  * finalizes the graph via `scaffold.finalize()`
  * create session
  * initializes the model via initialization ops provided by `Scaffold`
  * restores variables if a checkpoint exists
  * launches queue runners

  Run: When `run()` is called, the hooked session does following things:

  * calls `hook.before_run()`
  * calls TensorFlow `session.run()` with merged fetches and feed_dict
  * calls `hook.after_run()`
  * returns result of `session.run()` asked by user

  Exit: At the `close()`, the hooked session does following things in order:

  * calls `hook.end()`
  * closes the queue runners and the session
  * suppresses `OutOfRange` error which indicates that all inputs have been
    processed if the `SingularMonitoredSession` is used as a context.

  @compatibility(TF2)
  This API is not compatible with eager execution and `tf.function`. To migrate
  to TF2, rewrite the code to be compatible with eager execution. Check the
  [migration
  guide](https://www.tensorflow.org/guide/migrate#1_replace_v1sessionrun_calls)
  on replacing `Session.run` calls. In Keras, session hooks can be replaced by
  Callbacks e.g. [logging hook notebook](
  https://github.com/tensorflow/docs/blob/master/site/en/guide/migrate/logging_stop_hook.ipynb)
  For more details please read [Better
  performance with tf.function](https://www.tensorflow.org/guide/function).
  @end_compatibility
  """

    def __init__(self, hooks=None, scaffold=None, master='', config=None, checkpoint_dir=None, stop_grace_period_secs=120, checkpoint_filename_with_path=None):
        """Creates a SingularMonitoredSession.

    Args:
      hooks: An iterable of `SessionRunHook' objects.
      scaffold: A `Scaffold` used for gathering or building supportive ops. If
        not specified a default one is created. It's used to finalize the graph.
      master: `String` representation of the TensorFlow master to use.
      config: `ConfigProto` proto used to configure the session.
      checkpoint_dir: A string.  Optional path to a directory where to restore
        variables.
      stop_grace_period_secs: Number of seconds given to threads to stop after
        `close()` has been called.
      checkpoint_filename_with_path: A string. Optional path to a checkpoint
        file from which to restore variables.
    """
        session_creator = ChiefSessionCreator(scaffold=scaffold, master=master, config=config, checkpoint_dir=checkpoint_dir, checkpoint_filename_with_path=checkpoint_filename_with_path)
        super(SingularMonitoredSession, self).__init__(session_creator, hooks, should_recover=False, stop_grace_period_secs=stop_grace_period_secs)

    def raw_session(self):
        """Returns underlying `TensorFlow.Session` object."""
        return self._tf_sess()