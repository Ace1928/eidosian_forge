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
@tf_export(v1=['train.MonitoredTrainingSession'])
def MonitoredTrainingSession(master='', is_chief=True, checkpoint_dir=None, scaffold=None, hooks=None, chief_only_hooks=None, save_checkpoint_secs=USE_DEFAULT, save_summaries_steps=USE_DEFAULT, save_summaries_secs=USE_DEFAULT, config=None, stop_grace_period_secs=120, log_step_count_steps=100, max_wait_secs=7200, save_checkpoint_steps=USE_DEFAULT, summary_dir=None, save_graph_def=True):
    """Creates a `MonitoredSession` for training.

  For a chief, this utility sets proper session initializer/restorer. It also
  creates hooks related to checkpoint and summary saving. For workers, this
  utility sets proper session creator which waits for the chief to
  initialize/restore. Please check `tf.compat.v1.train.MonitoredSession` for
  more
  information.

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

  Args:
    master: `String` the TensorFlow master to use.
    is_chief: If `True`, it will take care of initialization and recovery the
      underlying TensorFlow session. If `False`, it will wait on a chief to
      initialize or recover the TensorFlow session.
    checkpoint_dir: A string.  Optional path to a directory where to restore
      variables.
    scaffold: A `Scaffold` used for gathering or building supportive ops. If not
      specified, a default one is created. It's used to finalize the graph.
    hooks: Optional list of `SessionRunHook` objects.
    chief_only_hooks: list of `SessionRunHook` objects. Activate these hooks if
      `is_chief==True`, ignore otherwise.
    save_checkpoint_secs: The frequency, in seconds, that a checkpoint is saved
      using a default checkpoint saver. If both `save_checkpoint_steps` and
      `save_checkpoint_secs` are set to `None`, then the default checkpoint
      saver isn't used. If both are provided, then only `save_checkpoint_secs`
      is used. Default 600.
    save_summaries_steps: The frequency, in number of global steps, that the
      summaries are written to disk using a default summary saver. If both
      `save_summaries_steps` and `save_summaries_secs` are set to `None`, then
      the default summary saver isn't used. Default 100.
    save_summaries_secs: The frequency, in secs, that the summaries are written
      to disk using a default summary saver.  If both `save_summaries_steps` and
      `save_summaries_secs` are set to `None`, then the default summary saver
      isn't used. Default not enabled.
    config: an instance of `tf.compat.v1.ConfigProto` proto used to configure
      the session. It's the `config` argument of constructor of
      `tf.compat.v1.Session`.
    stop_grace_period_secs: Number of seconds given to threads to stop after
      `close()` has been called.
    log_step_count_steps: The frequency, in number of global steps, that the
      global step/sec is logged.
    max_wait_secs: Maximum time workers should wait for the session to become
      available. This should be kept relatively short to help detect incorrect
      code, but sometimes may need to be increased if the chief takes a while to
      start up.
    save_checkpoint_steps: The frequency, in number of global steps, that a
      checkpoint is saved using a default checkpoint saver. If both
      `save_checkpoint_steps` and `save_checkpoint_secs` are set to `None`, then
      the default checkpoint saver isn't used. If both are provided, then only
      `save_checkpoint_secs` is used. Default not enabled.
    summary_dir: A string.  Optional path to a directory where to save
      summaries. If None, checkpoint_dir is used instead.
    save_graph_def: Whether to save the GraphDef and MetaGraphDef to
      `checkpoint_dir`. The GraphDef is saved after the session is created as
      `graph.pbtxt`. MetaGraphDefs are saved out for every checkpoint as
      `model.ckpt-*.meta`.

  Returns:
    A `MonitoredSession` object.
  """
    if save_summaries_steps == USE_DEFAULT and save_summaries_secs == USE_DEFAULT:
        save_summaries_steps = 100
        save_summaries_secs = None
    elif save_summaries_secs == USE_DEFAULT:
        save_summaries_secs = None
    elif save_summaries_steps == USE_DEFAULT:
        save_summaries_steps = None
    if save_checkpoint_steps == USE_DEFAULT and save_checkpoint_secs == USE_DEFAULT:
        save_checkpoint_steps = None
        save_checkpoint_secs = 600
    elif save_checkpoint_secs == USE_DEFAULT:
        save_checkpoint_secs = None
    elif save_checkpoint_steps == USE_DEFAULT:
        save_checkpoint_steps = None
    scaffold = scaffold or Scaffold()
    worker_context = distribute_coordinator_context.get_current_worker_context()
    if worker_context:
        return _create_monitored_session_with_worker_context(worker_context, scaffold, checkpoint_dir=checkpoint_dir, hooks=hooks, chief_only_hooks=chief_only_hooks, save_checkpoint_secs=save_checkpoint_secs, save_summaries_steps=save_summaries_steps, save_summaries_secs=save_summaries_secs, config=config, stop_grace_period_secs=stop_grace_period_secs, log_step_count_steps=log_step_count_steps, max_wait_secs=max_wait_secs, save_checkpoint_steps=save_checkpoint_steps, summary_dir=summary_dir, save_graph_def=save_graph_def)
    if not is_chief:
        session_creator = WorkerSessionCreator(scaffold=scaffold, master=master, config=config, max_wait_secs=max_wait_secs)
        return MonitoredSession(session_creator=session_creator, hooks=hooks or [], stop_grace_period_secs=stop_grace_period_secs)
    all_hooks = []
    if chief_only_hooks:
        all_hooks.extend(chief_only_hooks)
    session_creator = ChiefSessionCreator(scaffold=scaffold, checkpoint_dir=checkpoint_dir, master=master, config=config)
    summary_dir = summary_dir or checkpoint_dir
    if summary_dir:
        if log_step_count_steps and log_step_count_steps > 0:
            all_hooks.append(basic_session_run_hooks.StepCounterHook(output_dir=summary_dir, every_n_steps=log_step_count_steps))
        if save_summaries_steps and save_summaries_steps > 0 or (save_summaries_secs and save_summaries_secs > 0):
            all_hooks.append(basic_session_run_hooks.SummarySaverHook(scaffold=scaffold, save_steps=save_summaries_steps, save_secs=save_summaries_secs, output_dir=summary_dir))
    if checkpoint_dir:
        if save_checkpoint_secs and save_checkpoint_secs > 0 or (save_checkpoint_steps and save_checkpoint_steps > 0):
            all_hooks.append(basic_session_run_hooks.CheckpointSaverHook(checkpoint_dir, save_steps=save_checkpoint_steps, save_secs=save_checkpoint_secs, scaffold=scaffold, save_graph_def=save_graph_def))
    if hooks:
        all_hooks.extend(hooks)
    return MonitoredSession(session_creator=session_creator, hooks=all_hooks, stop_grace_period_secs=stop_grace_period_secs)