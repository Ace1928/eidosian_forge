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
class Cluster(object):
    """A cluster with workers.

  We assume all function errors are fatal and based on this assumption our
  error reporting logic is:
  1) Both `schedule` and `join` can raise a non-retryable error which is the
  first error seen by the coordinator from any previously scheduled functions.
  2) When an error is raised, there is no guarantee on how many previously
  scheduled functions have been executed; functions that have not been executed
  will be thrown away and marked as cancelled.
  3) After an error is raised, the internal state of error will be cleared.
  I.e. functions can continue to be scheduled and subsequent calls of `schedule`
  or `join` will not raise the same error again.

  Attributes:
    failure_handler: The failure handler used to handler worker preemption
      failure.
    workers: a list of `Worker` objects in the cluster.
    closure_queue: the global Closure queue.
    resource_cancellation_mgr: the cancellation manager used to cancel resource
      closures.
  """

    def __init__(self, strategy):
        """Initializes the cluster instance."""
        self._num_workers = strategy._num_workers
        self._num_ps = strategy._num_ps
        self._transient_ps_failures_threshold = int(os.environ.get('TF_COORDINATOR_IGNORE_TRANSIENT_PS_FAILURES', 3))
        self._potential_ps_failures_lock = threading.Lock()
        self._potential_ps_failures_count = [0] * self._num_ps
        self._transient_timeouts_threshold = int(os.environ.get('TF_COORDINATOR_IGNORE_TRANSIENT_TIMEOUTS', self._num_workers // 10))
        self._transient_timeouts_lock = threading.Lock()
        self._transient_timeouts_count = 0
        self.closure_queue = _CoordinatedClosureQueue()
        if os.getenv('TF_PSS_ENABLE_COORDINATION_SERVICE'):
            self.failure_handler = CoordinationServicePreemptionHandler(context.get_server_def(), self)
        else:
            self.failure_handler = WorkerPreemptionHandler(context.get_server_def(), self)
        worker_device_strings = ['/job:worker/replica:0/task:%d' % i for i in range(self._num_workers)]
        self.workers = [Worker(i, w, self) for i, w in enumerate(worker_device_strings)]
        self.resource_cancellation_mgr = cancellation.CancellationManager()

    def stop(self):
        """Stop worker, worker preemption threads, and the closure queue."""
        logging.info('Stopping cluster, starting with failure handler')
        self.failure_handler.stop()
        logging.info('Stopping workers')
        for worker in self.workers:
            worker.stop()
        logging.info('Stopping queue')
        self.closure_queue.stop()
        logging.info('Start cancelling remote resource-building functions')
        self.resource_cancellation_mgr.start_cancel()

    def _record_and_ignore_transient_ps_failure(self, e):
        """Records potential PS failures and return if failure should be ignored."""
        if self._transient_ps_failures_threshold <= 0 or not _is_ps_failure(e):
            return False
        ps_tasks = _extract_failed_ps_instances(str(e))
        with self._potential_ps_failures_lock:
            for t in ps_tasks:
                self._potential_ps_failures_count[t] += 1
                if self._potential_ps_failures_count[t] >= self._transient_ps_failures_threshold:
                    return False
        return True

    def _record_and_ignore_transient_timeouts(self, e):
        """Records observed timeout error and return if it should be ignored."""
        if self._transient_timeouts_threshold <= 0:
            return False
        if not isinstance(e, errors.DeadlineExceededError):
            return False
        with self._transient_timeouts_lock:
            self._transient_timeouts_count += 1
            if self._transient_timeouts_count >= self._transient_timeouts_threshold:
                return False
        return True

    def schedule(self, function, args, kwargs):
        """Schedules `function` to be dispatched to a worker for execution.

    Args:
      function: The function to be dispatched to a worker for execution
        asynchronously.
      args: Positional arguments for `fn`.
      kwargs: Keyword arguments for `fn`.

    Returns:
      A `RemoteValue` object.
    """
        closure = Closure(function, self.closure_queue._cancellation_mgr, args=args, kwargs=kwargs)
        ret = closure.build_output_remote_value()
        self.closure_queue.put(closure)
        return ret

    def join(self):
        """Blocks until all scheduled functions are executed."""
        self.closure_queue.wait()

    def done(self):
        """Returns true if all scheduled functions are executed."""
        return self.closure_queue.done()