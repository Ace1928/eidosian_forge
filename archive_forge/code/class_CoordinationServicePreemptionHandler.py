import collections
import contextlib
import os
import re
import threading
import time
import weakref
from six.moves import queue
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.distribute.coordinator import metric_utils
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.distribute.coordinator import utils
from tensorflow.python.distribute.coordinator import values as values_lib
from tensorflow.python.distribute.coordinator import watchdog
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
class CoordinationServicePreemptionHandler(object):
    """Handles preemptions of workers and parameter servers.

  Starts a thread to regularly poll the coordination service (hosted on PS 0)
  for task states. When a worker's task state reflects an error, it inspects the
  error. If the error is recoverable (i.e. a preemption), it waits for the
  worker to recover, then updates the server def. Otherwise, it raises the error
  to the user.

  A worker error is detected to be recoverable if it is the result of missing a
  heartbeat that workers regularly send to the coordination service.

  The thread also checks for parameter server errors. If these are detected, the
  thread and coordinator shutdown. To resume training in this case, the whole
  job must be restarted and resumed from the latest checkpoint.
  """

    def __init__(self, server_def, cluster):
        self._server_def = server_def
        self._cluster = cluster
        self._cluster_update_lock = threading.Lock()
        self._cluster_due_for_update_or_finish = threading.Event()
        self._worker_up_cond = threading.Condition(self._cluster_update_lock)
        self._next_task_state_cond = threading.Condition()
        self._task_states = None
        self._error_from_recovery = None
        self._should_preemption_thread_run = True
        self._task_state_poller_thread = utils.RepeatedTimer(interval=_POLL_FREQ_IN_SEC, function=self._get_task_states)
        self._preemption_handler_thread = threading.Thread(target=self._preemption_handler, name='WorkerPreemptionHandler', daemon=True)
        self._preemption_handler_thread.start()
        self._num_workers = self._cluster._num_workers
        self._num_ps = self._cluster._num_ps

    def stop(self):
        """Ensure the worker preemption thread is closed."""
        self._task_state_poller_thread.stop()
        self._should_preemption_thread_run = False
        with self._cluster_update_lock:
            self._cluster_due_for_update_or_finish.set()

    @contextlib.contextmanager
    def wait_on_failure(self, on_failure_fn=None, on_transient_failure_fn=None, on_recovery_fn=None, worker_device_name='(unknown)'):
        """Catches errors during closure execution and handles them.

    Args:
      on_failure_fn: an optional function to run if preemption happens.
      on_transient_failure_fn: an optional function to run if transient failure
        happens.
      on_recovery_fn: an optional function to run when a worker is recovered
        from preemption.
      worker_device_name: the device name of the worker instance that is passing
        through the failure.

    Yields:
      None.
    """
        assert self._should_preemption_thread_run
        try:
            yield
        except (errors.OpError, ClosureInputError, ClosureAbortedError) as e:
            with self._next_task_state_cond:
                self._next_task_state_cond.wait(_POLL_FREQ_IN_SEC * 1.25)
            with self._next_task_state_cond:
                self._next_task_state_cond.wait(_POLL_FREQ_IN_SEC * 1.25)
            if not self._task_states:
                self._log_ps_failure_and_raise(e, 0)
            worker_states = self._task_states[:self._num_workers]
            ps_states = self._task_states[self._num_workers:]
            if any(ps_states):
                failed_ps_index = [ix for ix, ps_state in enumerate(ps_states) if ps_state]
                self._log_ps_failure_and_raise(e, failed_ps_index[0])
            worker_ix = int(worker_device_name.split(':')[-1])
            if worker_states[worker_ix]:
                if self._cluster.closure_queue._cancellation_mgr.is_cancelled:
                    if isinstance(e, errors.CancelledError):
                        raise e
                    else:
                        raise errors.CancelledError(None, None, 'The corresponding function was cancelled while attempting to recover from worker failure.')
                self._handle_failure_and_recovery(e, on_failure_fn, on_transient_failure_fn, on_recovery_fn, worker_device_name)
                return
            if self._cluster._record_and_ignore_transient_timeouts(e):
                logging.error('Remote function on worker %s failed with %r:%s\nThis derived error is ignored and not reported to users.', worker_device_name, e, e)
                if on_transient_failure_fn:
                    on_transient_failure_fn()
                return
            raise e

    def _handle_failure_and_recovery(self, e, on_failure_fn, on_transient_failure_fn, on_recovery_fn, worker_device_name):
        """Call failure fn, wait for cluster to recover, then call recovery fn.

    Args:
      e: the Exception thrown during closure execution.
      on_failure_fn: an optional function to run if preemption happens.
      on_transient_failure_fn: an optional function to run if transient failure
        happens.
      on_recovery_fn: an optional function to run when a worker is recovered
        from preemption.
      worker_device_name: the device name of the worker instance that is passing
        through the failure.
    """
        if on_failure_fn:
            on_failure_fn(e)
        with self._cluster_update_lock:
            self._cluster_due_for_update_or_finish.set()
            self._worker_up_cond.wait(_WORKER_MAXIMUM_RECOVERY_SEC)
            if self._error_from_recovery:
                try:
                    raise self._error_from_recovery
                finally:
                    self._error_from_recovery = None
            logging.info('Worker %s has been recovered.', worker_device_name)
        if on_recovery_fn:
            logging.info('Worker %s calling on_recovery_fn', worker_device_name)
            with self.wait_on_failure(on_recovery_fn=on_recovery_fn, on_transient_failure_fn=on_transient_failure_fn, worker_device_name=worker_device_name):
                on_recovery_fn()

    def _log_ps_failure_and_raise(self, e, ps_index):
        logging.info('Parameter server failure detected at PS task %d', ps_index)
        self.stop()
        raise PSUnavailableError(e)

    def _get_task_states(self):
        """Get task states and reset to None if coordination service is down."""
        try:
            self._task_states = context.context().get_task_states([('worker', self._num_workers), ('ps', self._num_ps)])
        except (errors.UnavailableError, errors.InternalError) as e:
            if isinstance(e, errors.InternalError) and 'coordination service is not enabled' not in str(e).lower():
                raise
            self._task_states = None
        with self._next_task_state_cond:
            self._next_task_state_cond.notify_all()

    def _preemption_handler(self):
        """A loop that handles preemption.

    This loop waits for signal of worker preemption and upon worker preemption,
    it waits until all workers are back and updates the cluster about the
    restarted workers.
    """
        assert self._should_preemption_thread_run
        while True:
            self._cluster_due_for_update_or_finish.wait()
            if not self._should_preemption_thread_run:
                logging.info('Stopping the failure handing thread.')
                break
            with self._cluster_update_lock:
                try:
                    logging.info('Cluster now being recovered.')
                    context.context().update_server_def(self._server_def)
                    logging.info('Cluster successfully recovered.')
                    self._notify_cluster_update()
                except Exception as e:
                    logging.info('Error occurred while updating server def: %s', e)
                    with self._next_task_state_cond:
                        self._next_task_state_cond.wait(_POLL_FREQ_IN_SEC * 2)
                    if not self._task_states:
                        self._error_from_recovery = e
                    else:
                        ps_states = self._task_states[self._num_workers:]
                        if any(ps_states):
                            self._error_from_recovery = e
                    self._notify_cluster_update()
                    logging.error('Cluster update failed with error: %s. Retrying...', e)

    def _notify_cluster_update(self):
        self._worker_up_cond.notify_all()
        if self._should_preemption_thread_run:
            self._cluster_due_for_update_or_finish.clear()