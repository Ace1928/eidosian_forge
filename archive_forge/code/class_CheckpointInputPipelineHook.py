from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import ops
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('data.experimental.CheckpointInputPipelineHook')
class CheckpointInputPipelineHook(session_run_hook.SessionRunHook):
    """Checkpoints input pipeline state every N steps or seconds.

  This hook saves the state of the iterators in the `Graph` so that when
  training is resumed the input pipeline continues from where it left off.
  This could potentially avoid overfitting in certain pipelines where the
  number of training steps per eval are small compared to the dataset
  size or if the training pipeline is pre-empted.

  Differences from `CheckpointSaverHook`:
  1. Saves only the input pipelines in the "iterators" collection and not the
     global variables or other saveable objects.
  2. Does not write the `GraphDef` and `MetaGraphDef` to the summary.

  Example of checkpointing the training pipeline:

  ```python
  est = tf.estimator.Estimator(model_fn)
  while True:
    est.train(
        train_input_fn,
        hooks=[tf.data.experimental.CheckpointInputPipelineHook(est)],
        steps=train_steps_per_eval)
    # Note: We do not pass the hook here.
    metrics = est.evaluate(eval_input_fn)
    if should_stop_the_training(metrics):
      break
  ```

  This hook should be used if the input pipeline state needs to be saved
  separate from the model checkpoint. Doing so may be useful for a few reasons:
  1. The input pipeline checkpoint may be large, if there are large shuffle
     or prefetch buffers for instance, and may bloat the checkpoint size.
  2. If the input pipeline is shared between training and validation, restoring
     the checkpoint during validation may override the validation input
     pipeline.

  For saving the input pipeline checkpoint alongside the model weights use
  `tf.data.experimental.make_saveable_from_iterator` directly to create a
  `SaveableObject` and add to the `SAVEABLE_OBJECTS` collection. Note, however,
  that you will need to be careful not to restore the training iterator during
  eval. You can do that by not adding the iterator to the SAVEABLE_OBJECTS
  collector when building the eval graph.
  """

    def __init__(self, estimator, external_state_policy=None):
        """Initializes a `CheckpointInputPipelineHook`.

    If the input pipeline depends on external state (e.g. seeds for
    RandomUniform) beyond the input pipeline, this hook would be unable to
    serialize and deserialize that state. If its acceptable to ignore that state
    change the external_state_policy argument to 'warn' or 'ignore'. For e.g.

    ```python
    est = tf.estimator.Estimator(model_fn)
    while True:
      est.train(
          train_input_fn,
          hooks=[tf.data.experimental.CheckpointInputPipelineHook(
              est, external_state_policy='warn')],
          steps=train_steps_per_eval)
      # Note: We do not pass the hook here.
      metrics = est.evaluate(eval_input_fn)
      if should_stop_the_training(metrics):
        break
    ```

    Args:
      estimator: Estimator.
      external_state_policy: A string that identifies how to handle input
        pipelines that depend on external state. Possible values are
        'ignore': The external state is silently ignored.
        'warn': The external state is ignored, logging a warning.
        'fail': The operation fails upon encountering external state.
        By default we set it to 'fail'.

    Raises:
      ValueError: One of `save_steps` or `save_secs` should be set.
      ValueError: At most one of saver or scaffold should be set.
      ValueError: If `external_state_policy` is not one of 'warn', 'ignore' or
        'fail'.
    """
        if external_state_policy is None:
            external_state_policy = 'fail'
        self._external_state_policy = _convert_external_state_policy_to_enum(external_state_policy)
        checkpoint_prefix = 'input'
        if estimator._config.num_worker_replicas > 1:
            suffix = '_{}_{}'.format(estimator._config.task_type, estimator._config.task_id)
            checkpoint_prefix += suffix
        self._checkpoint_saver_hook = basic_session_run_hooks.CheckpointSaverHook(estimator.model_dir, save_secs=estimator._config.save_checkpoints_secs, save_steps=estimator._config.save_checkpoints_steps, checkpoint_basename=checkpoint_prefix + '.ckpt')
        self._latest_filename = 'checkpoint_' + checkpoint_prefix

    def begin(self):
        if self._checkpoint_saver_hook._saver is None and self._checkpoint_saver_hook._scaffold is None:
            iterators = ops.get_collection(iterator_ops.GLOBAL_ITERATORS)
            saveables = [iterator_ops._IteratorSaveable(i, i.name, external_state_policy=self._external_state_policy) for i in iterators]
            self._checkpoint_saver_hook._saver = _CustomSaver(saveables, self._latest_filename, sharded=True)
        self._checkpoint_saver_hook.begin()

    def after_create_session(self, session, coord):
        self._first_run = True

    def _restore_or_save_initial_ckpt(self, session):
        latest_checkpoint_path = checkpoint_management.latest_checkpoint(self._checkpoint_saver_hook._checkpoint_dir, latest_filename=self._latest_filename)
        if latest_checkpoint_path:
            self._checkpoint_saver_hook._get_saver().restore(session, latest_checkpoint_path)
        else:
            global_step = session.run(self._checkpoint_saver_hook._global_step_tensor)
            self._checkpoint_saver_hook._save(session, global_step)
            self._checkpoint_saver_hook._timer.update_last_triggered_step(global_step)

    def before_run(self, run_context):
        if self._first_run:
            self._restore_or_save_initial_ckpt(run_context.session)
            self._first_run = False
        return self._checkpoint_saver_hook.before_run(run_context)

    def after_run(self, run_context, run_values):
        self._checkpoint_saver_hook.after_run(run_context, run_values)

    def end(self, session):
        self._checkpoint_saver_hook.end(session)