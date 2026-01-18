import os
import signal
import sys
import threading
import time
from tensorflow.core.distributed_runtime.preemption import gen_check_preemption_op
from tensorflow.python.checkpoint import checkpoint as checkpoint_lib
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute.failure_handling import failure_handling_util
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@tf_export('distribute.experimental.PreemptionCheckpointHandler', v1=[])
class PreemptionCheckpointHandler(object):
    """Preemption and error handler for synchronous training.

  Note: This API only supports use with
  `tf.distribute.MultiWorkerMirroredStrategy` and `tf.distribute.TPUStrategy`.

  A `PreemptionCheckpointHandler` coordinates all workers to save a checkpoint
  upon receiving a preemption signal. It also helps disseminate application
  error messages accurately among the cluster. When a
  `PreemptionCheckpointHandler` object is created, it restores values from
  the latest checkpoint file if any exists.

  Right after the initialization, the object starts to watch out for termination
  signal for any member in the cluster. If receiving a signal, the next time the
  worker executes `PreemptionCheckpointHandler.run`, the
  `PreemptionCheckpointHandler` will align all workers to save a checkpoint.
  Then, if an `exit_fn` is configured via
  `tf.distribute.experimental.TerminationConfig`, it will be invoked. Otherwise,
  the process will simply exit and later the platform should restart it.

  Note: We advise users of `tf.distribute.MultiWorkerMirroredStrategy` who
  choose to configure their
  own `exit_fn` in `tf.distribute.experimental.TerminationConfig` to include a
  `sys.exit(CODE_OR_MESSAGE)` in the `exit_fn` so that after the restart, all
  workers can initialize communication services correctly. For users of
  `tf.distribute.TPUStrategy`, if they do not wish to do a cluster restart but
  would like an in-process restart (i.e., keep the coordinator alive and re-do
  the steps to connect to cluster, initialize TPU system, and make the
  `TPUStrategy` object), they could configure the `exit_fn` to a no-op.

  For users of `tf.distribute.MultiWorkerMirroredStrategy`, the core API is
  `PreemptionCheckpointHandler.run`:

  ```python
  strategy = tf.distribute.MultiWorkerMirroredStrategy()

  trained_epoch = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='epoch')
  step_in_epoch = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='step_in_epoch')

  with strategy.scope():
    dataset, model, optimizer = ...

    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=model,
                                     trained_epoch=trained_epoch,
                                     step_in_epoch=step_in_epoch)

    preemption_checkpoint_handler = tf.distribute.experimental.PreemptionCheckpointHandler(cluster_resolver, checkpoint, checkpoint_dir)

  while trained_epoch.numpy() < NUM_EPOCH:

    while step_in_epoch.numpy() < STEPS_PER_EPOCH:

      # distributed_train_function contains a call to strategy.run.
      loss += preemption_checkpoint_handler.run(distributed_train_function, args=(next(iterator),))
      # For users of MultiWorkerMirroredStrategy, usually
      # STEPS_PER_TRAIN_FUNCTION = 1.
      step_in_epoch.assign_add(STEPS_PER_TRAIN_FUNCTION)
      ...

    epoch.assign_add(1)
    step_in_epoch.assign(0)
  ```

  For users of `tf.distribute.TPUStrategy`, the core APIs are
  `PreemptionCheckpointHandler.run` and
  `PreemptionCheckpointHandler.watch_preemption_scope`:

  ```python

  strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)

  # Rest of TPU init omitted, see documentation for TPUSTrategy.

  with preemption_checkpoint_handler.watch_preemption_scope():
    while trained_epoch.numpy() < NUM_EPOCH:

      while step_in_epoch.numpy() < STEPS_PER_EPOCH:

        # distributed_train_function contains a call to strategy.run.
        loss += preemption_checkpoint_handler.run(distributed_train_function, args=(next(iterator),))

        # For users of TPUStrategy, usually STEPS_PER_TRAIN_FUNCTION >> 1 since
        # clustering multiple steps within a tf.function amortizes the overhead
        # of launching a multi-device function on TPU Pod.
        step_in_epoch.assign_add(STEPS_PER_TRAIN_FUNCTION)
        ...

      epoch.assign_add(1)
      step_in_epoch.assign(0)
  ```

  Not all interruptions come with advance notice so that the
  `PreemptionCheckpointHandler` can handle them, e.g., those caused by hardware
  failure. For a user who saves checkpoints for these cases themselves outside
  the `PreemptionCheckpointHandler`, if they are using a
  `tf.train.CheckpointManager`, pass it as the
  `checkpoint_or_checkpoint_manager` argument to the
  `PreemptionCheckpointHandler`. If they do not have a
  `tf.train.CheckpointManager` but are directly working with
  `tf.train.Checkpoint`, we advise saving the checkpoints in the directory
  that's passed as the `checkpoint_dir` argument. In this way, at the program
  beginning, `PreemptionCheckpointHandler` can restore the latest checkpoint
  from the directory, no matter it's saved by the user themselves or saved by
  the `PreemptionCheckpointHandler` before preemption happens.

  **A note on the platform:**

  `PreemptionCheckpointHandler` can only handle the kind of termination with
  advance notice. For now, the API recognizes the termination signal for CPU,
  GPU, and TPU on Google Borg and CPU and GPU on the Google Cloud Platform. In
  these cases, `PreemptionCheckpointHandler` will automatically adopt the
  correct preemption/maintenance notification detection mechanism. Users of
  other platforms can configure a detection monitoring behavior through the
  `tf.distribute.experimental.TerminationConfig`. Customization for the exit
  behavior and grace period length could also be done here.
  """

    def __init__(self, cluster_resolver, checkpoint_or_checkpoint_manager, checkpoint_dir=None, termination_config=None):
        """Creates the `PreemptionCheckpointHandler`.

    Args:
      cluster_resolver: a `tf.distribute.cluster_resolver.ClusterResolver`
        object. You may also obtain it through the `cluster_resolver` attribute
        of the distribution strategy in use.
      checkpoint_or_checkpoint_manager: a `tf.train.CheckpointManager` or a
        `tf.train.Checkpoint`. If you are using a `tf.train.CheckpointManager`
        to manage checkpoints outside the `PreemptionCheckpointHandler` for
        backup purpose as well, pass it as `checkpoint_or_checkpoint_manager`
        argument. Otherwise, pass a `tf.train.Checkpoint` and the
        `PreemptionCheckpointHandler` will create
        a `tf.train.CheckpointManager` to manage it in the `checkpoint_dir`.
      checkpoint_dir: a directory where the `PreemptionCheckpointHandler` saves
        and restores checkpoints. When a `PreemptionCheckpointHandler` is
        created, the latest checkpoint in the `checkpoint_dir` will be restored.
        (This is not needed if a `tf.train.CheckpointManager` instead of a
        `tf.train.Checkpoint` is passed as the
        `checkpoint_or_checkpoint_manager` argument.)
      termination_config: optional, a
        `tf.distribute.experimental.TerminationConfig` object to configure for a
        platform other than Google Borg or GCP.
    """
        if isinstance(checkpoint_or_checkpoint_manager, checkpoint_lib.Checkpoint) and (not checkpoint_dir):
            raise errors.InvalidArgumentError('When a checkpoint is passed, a checkpoint_dir must be passed as well.')
        self._cluster_resolver = cluster_resolver
        self._termination_config = termination_config
        self._checkpoint_or_checkpoint_manager = checkpoint_or_checkpoint_manager
        self._checkpoint_dir = checkpoint_dir
        self._platform_device = failure_handling_util.detect_platform()
        completed_termination_config = _complete_config_for_environment(self._platform_device, self._termination_config)
        self._termination_watcher_fn = completed_termination_config.termination_watcher_fn
        self._exit_fn = completed_termination_config.exit_fn
        self._grace_period = completed_termination_config.grace_period
        self._save_fn = completed_termination_config.save_fn
        self._local_mode = True
        if self._platform_device in (failure_handling_util.PlatformDevice.GCE_TPU, failure_handling_util.PlatformDevice.GCE_CPU):
            logging.warning('PreemptionCheckpointHandler does not support usage with TPU or CPU device on GCP.')
        elif self._platform_device == failure_handling_util.PlatformDevice.INTERNAL_TPU:
            self._initialize_for_tpu_strategy()
        else:
            if cluster_resolver and 'ps' in cluster_resolver.cluster_spec().as_dict():
                raise NotImplementedError('PreemptionCheckpointHandler does not supportusage with tf.distribute.experimental.ParameterServerStrategy.')
            self._initialize_for_mirrored_and_multi_worker_mirrored()
        logging.info('PreemptionCheckpointHandler initialized or restored.')

    def _initialize_for_tpu_strategy(self):
        """Makes configurations for using the handler with TPUStrategy."""
        self._is_chief = True
        self._poll_termination_signal_thread = None
        self._cluster_wise_termination_watcher_thread = None
        self._maybe_create_checkpoint_manager()
        self._read_checkpoint_manager.restore_or_initialize()
        self._run_counter = 0

    def _initialize_for_mirrored_and_multi_worker_mirrored(self):
        """Makes configurations and start watchers for MS, MWMS, or OneDevice."""
        if not self._cluster_resolver or not self._cluster_resolver.cluster_spec().jobs:
            self._local_mode = True
            self._id_in_cluster = 'single_worker'
            self._is_chief = True
        else:
            self._local_mode = False
            self._id_in_cluster = str(multi_worker_util.id_in_cluster(self._cluster_resolver.cluster_spec(), self._cluster_resolver.task_type, self._cluster_resolver.task_id))
            self._is_chief = multi_worker_util.is_chief(cluster_spec=self._cluster_resolver.cluster_spec(), task_type=self._cluster_resolver.task_type, task_id=self._cluster_resolver.task_id)
        self._checkpointed_runs = variables.Variable(initial_value=constant_op.constant(0, dtype=dtypes.int64), trainable=False, name=_ITERATION_VARIABLE)
        self._maybe_create_checkpoint_manager()
        if not hasattr(self._write_checkpoint_manager._checkpoint, _ITERATION_VARIABLE):
            setattr(self._write_checkpoint_manager._checkpoint, _ITERATION_VARIABLE, self._checkpointed_runs)
        if not hasattr(self._read_checkpoint_manager._checkpoint, _ITERATION_VARIABLE):
            setattr(self._read_checkpoint_manager._checkpoint, _ITERATION_VARIABLE, self._checkpointed_runs)
        self._read_checkpoint_manager.restore_or_initialize()
        self._final_checkpoint_countdown = False
        self._estimated_run_time = 0
        self._run_counter = self._checkpointed_runs.numpy()
        self._received_own_sigterm = threading.Event()
        self._received_checkpoint_step = threading.Event()
        distribute_lib.distribution_strategy_input_api_counter.get_cell(self._platform_device.name, 'PreemptionCheckpointHandler').increase_by(1)
        if not self._local_mode:
            self._cluster_wise_termination_watcher_thread = threading.Thread(target=self._watch_step_to_save_key, name='PeerTerminationWatcher-%s' % self._id_in_cluster, daemon=True)
            logging.info("Start watcher for peer's signal.")
            self._cluster_wise_termination_watcher_thread.start()
        else:
            self._cluster_wise_termination_watcher_thread = None
        self._poll_termination_signal_thread = None
        if self._termination_watcher_fn:
            self._start_polling_for_termination_signal()
        else:
            self._start_watching_for_signal()

    def _maybe_create_checkpoint_manager(self):
        """Create CheckpointManager(s) if a checkpoint is passed else take it."""
        if isinstance(self._checkpoint_or_checkpoint_manager, checkpoint_management.CheckpointManager):
            self._read_checkpoint_manager = self._checkpoint_or_checkpoint_manager
            self._write_checkpoint_manager = self._checkpoint_or_checkpoint_manager
            self._api_made_checkpoint_manager = False
        else:
            self._api_made_checkpoint_manager = True
            self._read_checkpoint_manager = checkpoint_management.CheckpointManager(self._checkpoint_or_checkpoint_manager, directory=self._checkpoint_dir, max_to_keep=1)
            if self._is_chief:
                self._write_checkpoint_manager = self._read_checkpoint_manager
            else:
                self._write_checkpoint_manager = checkpoint_management.CheckpointManager(self._checkpoint_or_checkpoint_manager, _non_chief_checkpoint_dir(self._checkpoint_dir, self._cluster_resolver.task_id), max_to_keep=1)

    def _start_watching_for_signal(self):
        logging.info('Start watcher for local signal.')
        signal.signal(signal.SIGTERM, self._sigterm_handler_fn)

    def _start_polling_for_termination_signal(self):
        self._poll_termination_signal_thread_should_stop = threading.Event()
        self._poll_termination_signal_thread = threading.Thread(target=self._poll_termination_signal, name='WorkerTerminationSignalWatcher-%s' % self._id_in_cluster, daemon=True)
        logging.info('Start polling for termination signal.')
        self._poll_termination_signal_thread.start()

    def _poll_termination_signal(self):
        """Poll maintenance notice and notify peers if receiving one."""
        while True:
            if self._poll_termination_signal_thread_should_stop.is_set() or self._final_checkpoint_countdown:
                return
            if self._termination_watcher_fn():
                break
            time.sleep(1)
        self._maybe_set_received_own_sigterm()

    def _maybe_set_received_own_sigterm(self):
        """Claim earliest preemption if no one else has done it before."""
        if self._local_mode:
            logging.info('Member %s has received termination notice.', self._id_in_cluster)
            self._received_own_sigterm_time = time.time()
            self._received_own_sigterm.set()
            return
        try:
            context.context().set_config_key_value(_PREEMPTION_WORKER_KEY, self._id_in_cluster)
            logging.info('Member %s has received termination notice.', self._id_in_cluster)
            self._received_own_sigterm_time = time.time()
            self._received_own_sigterm.set()
        except errors.AlreadyExistsError:
            logging.info('Member %s has received termination notice. But some other worker has received it as well! Leaving it to them to decide when to checkpoint. ', self._id_in_cluster)
            return

    def _stop_poll_termination_signal_thread(self):
        if getattr(self, '_poll_termination_signal_thread', None):
            self._poll_termination_signal_thread_should_stop.set()
            self._poll_termination_signal_thread.join()
            self._poll_termination_signal_thread = None
            logging.info("Shut down watcher for one's own termination signal")

    def _stop_cluster_wise_termination_watcher_thread(self):
        """Stop the thread that is _watch_step_to_save_key."""
        if getattr(self, '_cluster_wise_termination_watcher_thread', None):
            try:
                context.context().set_config_key_value(_INITIAL_RUN_COUNT_KEY, _STOP_WATCHING_CLUSTER_VALUE)
            except (errors.AlreadyExistsError, errors.UnavailableError):
                pass
            except Exception as e:
                logging.info('Ignoring error when shutting down _stop_cluster_wise_termination_watcher_thread: ' + str(e))
            try:
                context.context().set_config_key_value(_FINAL_RUN_COUNT_KEY, _STOP_WATCHING_CLUSTER_VALUE)
            except (errors.AlreadyExistsError, errors.UnavailableError):
                pass
            except Exception as e:
                logging.info('Ignoring error when shutting down _stop_cluster_wise_termination_watcher_thread: ' + str(e))
            finally:
                self._cluster_wise_termination_watcher_thread.join()
                self._cluster_wise_termination_watcher_thread = None
                logging.info("Shut down watcher for peer's termination signal.")

    def __del__(self):
        self._stop_cluster_wise_termination_watcher_thread()
        self._stop_poll_termination_signal_thread()

    @property
    @deprecated(None, 'Track steps using a tf.Variable saved in checkpoint instead.')
    @doc_controls.do_not_generate_docs
    def total_run_calls(self):
        """Returns the number of times `PreemptionCheckpointHandler.run` is called.

    DEPRECATED: user should track total steps themselves, as this API provides
    little expressivity gain but could easily be misused and incurs extra
    synchronization cost for TPUStrategy users.

    This value tracks the number of all calls to
    `PreemptionCheckpointHandler.run` including those before the program is
    restarted and the training is restored, by saving and reading the value in
    the checkpoint. A user can compute their total number of iterations
    by `PreemptionCheckpointHandler.total_run_calls *
    number_of_steps_in_train_function`,
    while `number_of_steps_in_train_function` should be one for
    `tf.distribute.MultiWorkerMirroredStrategy` users. They can also use this
    value to infer the starting epoch and step after training restores, as shown
    in the example above.
    """
        if self._platform_device == failure_handling_util.PlatformDevice.INTERNAL_TPU:
            raise NotImplementedError('Please create variables saved in checkpoint to keep track of steps and epochs.')
        return self._run_counter

    def run(self, distributed_train_function, *args, **kwargs):
        """Runs a training function with error and preemption handling.

    This function handles the preemption signal from any peer in the cluster by
    saving the training progress and exiting gracefully. It will
    also broadcase any program error encountered during the execution of
    `distributed_train_function` to all workers so that they can raise the same
    error.

    The `distributed_train_function` argument should be a distributed train
    function (i.e., containing a call to `tf.distribute.Strategy.run`). For
    `tf.distribute.MultiWorkerMirroredStrategy` users, we recommend passing in a
    single-step `distributed_train_function` to
    `PreemptionCheckpointHandler.run` so that the checkpoint can be saved in
    time in case a preemption signal or maintenance notice is sent.

    Besides the preemption and error handling part,
    `PreemptionCheckpointHandler.run(distributed_train_function, *args,
    **kwargs)` has the same effect and output as
    `distributed_train_function(*args, **kwargs)`. `distributed_train_function`
    can return either some or no result. The following is a shortened example:

    ```python

    @tf.function
    def distributed_train_step(iterator):
      # A distributed single-step training function.

      def step_fn(inputs):
        # A per-replica single-step training function.
        x, y = inputs
        ...
        return loss

      per_replica_losses = strategy.run(step_fn, args=(next(iterator),))
      return strategy.reduce(
          tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    for epoch in range(preemption_handler.total_run_calls // STEPS_PER_EPOCH,
                       EPOCHS_TO_RUN):
      iterator = iter(multi_worker_dataset)
      total_loss = 0.0
      num_batches = 0

      for step in range(preemption_handler.total_run_calls % STEPS_PER_EPOCH,
                        STEPS_PER_EPOCH):
        total_loss += preemption_handler.run(distributed_train_step)
        num_batches += 1

      train_loss = total_loss / num_batches
      print('Epoch: %d, train_loss: %f.' %(epoch.numpy(), train_loss))

      train_accuracy.reset_states()
    ```

    Args:
      distributed_train_function: A (single-step) distributed training function.
      *args: args for `distributed_train_function`.
      **kwargs: kwargs for `distributed_train_function`.

    Raises:
      Program error encountered by any member in the cluster while executing the
      `distributed_train_function`, or any error from the program error
      propagation process.

    Returns:
      Result of running the `distributed_train_function`.
    """
        if self._platform_device == failure_handling_util.PlatformDevice.INTERNAL_TPU:
            return self._run_for_tpu(distributed_train_function, *args, **kwargs)
        elif self._platform_device in (failure_handling_util.PlatformDevice.GCE_TPU, failure_handling_util.PlatformDevice.GCE_CPU):
            return distributed_train_function(*args, **kwargs)
        else:
            return self._run_for_multi_worker_mirrored(distributed_train_function, *args, **kwargs)

    def _run_for_tpu(self, distributed_train_function, *args, **kwargs):
        """PreemptionCheckpointHandler.run implementation for TPUStrategy."""
        gen_check_preemption_op.check_preemption(preemption_key=PREEMPTION_KEY)
        return distributed_train_function(*args, **kwargs)

    def _run_for_multi_worker_mirrored(self, distributed_train_function, *args, **kwargs):
        """PreemptionCheckpointHandler.run implementation for MWMS."""
        try:
            self._check_preemption_and_maybe_checkpoint()
            run_begin_time = time.time()
            result = distributed_train_function(*args, **kwargs)
            new_run_time = time.time() - run_begin_time
            self._run_counter += 1
            self._estimated_run_time = self._estimated_run_time + (new_run_time - self._estimated_run_time) / self._run_counter
        except errors.OpError as e:
            if not self._local_mode:
                logging.info('Propagating error to cluster: %r: %s', e, e)
                try:
                    context.context().report_error_to_cluster(e.error_code, e.message)
                except Exception as ex:
                    logging.info('Ignoring error during error propagation: %r:%s', ex, ex)
            raise
        return result

    def save_checkpoint_if_preempted(self, *args, **kwargs):
        """Saves a checkpoint if a preemption signal has been made available.

    This is an alternative API for `PreemptionCheckpointHandler.run` and
    `PreemptionCheckpointHandler.watch_preemption_scope`. This method works for
    both `tf.distribute.MultiWorkerMirroredStrategy` and
    `tf.distribute.TPUStrategy`. However, **for TPUStrategy, this method will
    add a synchronization point between workers and the coordinator** and thus
    may have performance implication. If this is a concern, use the combination
    of `PreemptionCheckpointHandler.watch_preemption_scope` and
    `PreemptionCheckpointHandler.run` instead.

    ```python
    strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)
    # initialization omitted

    with strategy.scope():
      # Save in the checkpoint.
      trained_step = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='trained_step', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

      checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory, max_to_keep=1)
      preemption_handler = tf.distribute.experimental.PreemptionCheckpointHandler(cluster_resolver, checkpoint_manager)

    while trained_step.numpy() < NUM_STEPS:
      # Train STEPS_IN_FUNCTION steps at once.
      train_multi_step_function()
      trained_step.assign_add(STEPS_IN_FUNCTION)
      preemption_handler.save_checkpoint_if_preempted()
    ```

    Args:
      *args: args for `tf.train.CheckpointManager.save()` to save checkpoint.
      **kwargs: kwargs for `tf.train.CheckpointManager.save()` to save.
    """
        if self._platform_device == failure_handling_util.PlatformDevice.INTERNAL_TPU:
            try:
                with context.async_scope():
                    gen_check_preemption_op.check_preemption(preemption_key=PREEMPTION_KEY)
            except errors.AbortedError as abort_error:
                if abort_error.experimental_payloads.get(b'type.googleapis.com/tensorflow.distributed_runtime.WorkerPreemption'):
                    logging.info('Clearing preemption error to save checkpoint...')
                    context.async_clear_error()
                    self._save_checkpoint(*args, **kwargs)
                    self._exit_fn()
                else:
                    raise
        elif self._platform_device in (failure_handling_util.PlatformDevice.GCE_TPU, failure_handling_util.PlatformDevice.GCE_CPU):
            return
        else:
            self._check_preemption_and_maybe_checkpoint(*args, **kwargs)
            self._run_counter += 1
            self._estimated_run_time = 0

    @tf_contextlib.contextmanager
    def watch_preemption_scope(self):
        """Syncs error and maybe save checkpoint for usage with TPUStrategy.

    Note: Usage with `tf.distribute.MultiWorkerMirroredStrategy` does not need
    this API.

    Example usage:

    ```python
    with preemption_checkpoint_handler.watch_preemption_scope():
      while trained_step.numpy() < NUM_STEPS:

        # distributed_train_function contains a call to strategy.run.
        loss += preemption_checkpoint_handler.run(distributed_train_function, args=(next(iterator),))
        trained_step.assign_add(STEPS_PER_TRAIN_FUNCTION)
    ```

    In this workflow, `PreemptionCheckpointHandler.run` will flag preemption
    signal received, and `watch_preemption_scope` will handle the preemption
    signal by saving a checkpoint and then either exit to restart or execute a
    user-passed `exit_fn` in `tf.distribute.experimental.TerminationConfig`. If
    no preemption signal is received during execution of ops and function inside
    the scope, `watch_preemption_scope` ensures the completion of all async op
    and function execution when exiting and will raises exceptions if async
    execution results in an error state.

    Yields:
      None
    """
        if self._platform_device == failure_handling_util.PlatformDevice.INTERNAL_TPU:
            try:
                with context.async_scope():
                    yield
            except errors.AbortedError as abort_error:
                if abort_error.experimental_payloads.get(b'type.googleapis.com/tensorflow.distributed_runtime.WorkerPreemption'):
                    logging.info('Clearing preemption error to save checkpoint...')
                    context.async_clear_error()
                    self._save_checkpoint()
                    self._exit_fn()
                else:
                    raise
        else:
            try:
                yield
            except errors.OpError as e:
                if not self._local_mode:
                    logging.info('Propagating error to cluster: %r: %s', e, e)
                    try:
                        context.context().report_error_to_cluster(e.error_code, e.message)
                    except Exception as ex:
                        logging.info('Ignoring error during error propagation: %r:%s', ex, ex)
                raise

    def _save_checkpoint(self, *args, **kwargs):
        """Saves the checkpoint and exit program."""
        distribute_lib.distribution_strategy_input_api_counter.get_cell(self._platform_device.name, 'PreemptionCheckpointHandler Saving Checkpoint').increase_by(1)
        logging.info('PreemptionCheckpointHandler: Starting saving a checkpoint.')
        if self._platform_device != failure_handling_util.PlatformDevice.INTERNAL_TPU:
            self._checkpointed_runs.assign(self.total_run_calls)
        start_time = time.monotonic()
        with checkpoint_context.preemption_save_context():
            if self._save_fn:
                self._save_fn(*args, **kwargs)
            else:
                self._write_checkpoint_manager.save(*args, **kwargs)
        end_time = time.monotonic()
        logging.info('Checkpoint finished at path %s', self._write_checkpoint_manager.directory)
        self._checkpoint_time = end_time - start_time

    def _check_preemption_and_maybe_checkpoint(self, *args, **kwargs):
        """Checkpoint if any worker has received a preemption signal.

    This function handles preemption signal reported by any worker in the
    cluster. The current implementation relies on the fact that all workers in a
    MultiWorkerMirroredStrategy training cluster have a step number difference
    maximum of 1.
    - If the signal comes from the worker itself (i.e., where this failure
    handler sits), the worker will notify all peers to checkpoint after they
    finish CURRENT_STEP+1 steps, where CURRENT_STEP is the step this worker has
    just finished. And the worker will wait for all peers to acknowledge that
    they have received its preemption signal and the final-step number before
    the worker proceeds on training the final step.
    - If the signal comes from another member in the cluster but NO final-step
    info is available, proceed on training, because it will be available after
    finishing the next step.
    - If the signal comes from some other member in the cluster, and final-step
    info is available, if the worker has not finished these steps yet, keep
    training; otherwise, checkpoint and exit with a cluster-recognized restart
    code.

    Args:
      *args: args for `tf.train.CheckpointManager.save()` to save checkpoint.
      **kwargs: kwargs for `tf.train.CheckpointManager.save()` to save.
    """
        if self._platform_device == failure_handling_util.PlatformDevice.INTERNAL_TPU:
            gen_check_preemption_op.check_preemption(preemption_key=PREEMPTION_KEY)
            return
        if self._final_checkpoint_countdown:
            run_count_config_key = _FINAL_RUN_COUNT_KEY
        else:
            run_count_config_key = _INITIAL_RUN_COUNT_KEY
        if self._received_checkpoint_step.is_set():
            if self._step_to_checkpoint == str(self._run_counter):
                self._save_checkpoint(*args, **kwargs)
                if self._time_to_exit():
                    self._stop_poll_termination_signal_thread()
                    self._stop_cluster_wise_termination_watcher_thread()
                    if self._api_made_checkpoint_manager and (not self._is_chief):
                        gfile.DeleteRecursively(os.path.dirname(self._write_checkpoint_manager.directory))
                    logging.info('PreemptionCheckpointHandler: checkpoint saved. Exiting.')
                    self._exit_fn()
                else:
                    logging.info('Continue training for the grace period.')
                    self._final_checkpoint_countdown = True
                    self._received_checkpoint_step.clear()
        elif self._received_own_sigterm.is_set():
            if self._final_checkpoint_countdown:
                if self._target_time_for_termination < time.time():
                    logging.info('Grace period almost ended. Final call to save a checkpoint!')
                else:
                    return
            step_to_save_at = str(self._run_counter + 1)
            logging.info('Termination caught in main thread on preempted worker')
            if self._local_mode:
                self._step_to_checkpoint = step_to_save_at
                self._received_checkpoint_step.set()
            else:
                context.context().set_config_key_value(run_count_config_key, step_to_save_at)
                logging.info('%s set to %s', run_count_config_key, step_to_save_at)
                if not self._local_mode:
                    worker_count = multi_worker_util.worker_count(self._cluster_resolver.cluster_spec(), self._cluster_resolver.task_type)
                    for i in range(worker_count):
                        context.context().get_config_key_value(f'{_ACKNOWLEDGE_KEY}_{run_count_config_key}_{i}')
                        logging.info('Sigterm acknowledgement from replica %d received', i)
            self._setup_countdown_if_has_grace_period_and_not_already_counting_down()

    def _time_to_exit(self):
        """Return whether to exit: exit if no grace period or grace period ends."""
        return self._grace_period <= 0 or self._final_checkpoint_countdown

    def _setup_countdown_if_has_grace_period_and_not_already_counting_down(self):
        """Set up at the beginning of a countdown period for long grace period."""
        if self._grace_period > 0 and (not self._final_checkpoint_countdown):
            buffer_factor = 3
            self._target_time_for_termination = self._received_own_sigterm_time + self._grace_period - buffer_factor * self._estimated_run_time * 2

    def _sigterm_handler_fn(self, signum, frame):
        """Upload the to-be-preempted worker's id to coordination service."""
        del signum, frame
        self._maybe_set_received_own_sigterm()

    def _watch_step_to_save_key(self):
        """Watch out for step-to-save config key and acknowledge.

    All workers, including the one to be preempted, execute this function to get
    step-to-save.
    """
        step_value = context.context().get_config_key_value(_INITIAL_RUN_COUNT_KEY)
        if step_value != _STOP_WATCHING_CLUSTER_VALUE:
            self._step_to_checkpoint = step_value
            self._received_checkpoint_step.set()
            ack_key = f'{_ACKNOWLEDGE_KEY}_{_INITIAL_RUN_COUNT_KEY}_{self._id_in_cluster}'
            context.context().set_config_key_value(ack_key, '1')
            logging.info('PreemptionCheckpointHandler: %s set, preemption awareness acknowledged', ack_key)
            if self._grace_period > 0:
                final_step_value = context.context().get_config_key_value(_FINAL_RUN_COUNT_KEY)
                if final_step_value != _STOP_WATCHING_CLUSTER_VALUE:
                    ack_key = f'{_ACKNOWLEDGE_KEY}_{_FINAL_RUN_COUNT_KEY}_{self._id_in_cluster}'
                    context.context().set_config_key_value(ack_key, '1')
                    logging.info('PreemptionCheckpointHandler: %s acknowledged, final checkpoint timing received.', ack_key)
                    self._received_checkpoint_step.set()
                    self._step_to_checkpoint = final_step_value