import threading
import weakref
from tensorflow.core.protobuf import queue_runner_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _init_from_args(self, queue=None, enqueue_ops=None, close_op=None, cancel_op=None, queue_closed_exception_types=None):
    """Create a QueueRunner from arguments.

    Args:
      queue: A `Queue`.
      enqueue_ops: List of enqueue ops to run in threads later.
      close_op: Op to close the queue. Pending enqueue ops are preserved.
      cancel_op: Op to close the queue and cancel pending enqueue ops.
      queue_closed_exception_types: Tuple of exception types, which indicate
        the queue has been safely closed.

    Raises:
      ValueError: If `queue` or `enqueue_ops` are not provided when not
        restoring from `queue_runner_def`.
      TypeError: If `queue_closed_exception_types` is provided, but is not
        a non-empty tuple of error types (subclasses of `tf.errors.OpError`).
    """
    if not queue or not enqueue_ops:
        raise ValueError('Must provide queue and enqueue_ops.')
    self._queue = queue
    self._enqueue_ops = enqueue_ops
    self._close_op = close_op
    self._cancel_op = cancel_op
    if queue_closed_exception_types is not None:
        if not isinstance(queue_closed_exception_types, tuple) or not queue_closed_exception_types or (not all((issubclass(t, errors.OpError) for t in queue_closed_exception_types))):
            raise TypeError('queue_closed_exception_types, when provided, must be a tuple of tf.error types, but saw: %s' % queue_closed_exception_types)
    self._queue_closed_exception_types = queue_closed_exception_types
    if self._close_op is None:
        self._close_op = self._queue.close()
    if self._cancel_op is None:
        self._cancel_op = self._queue.close(cancel_pending_enqueues=True)
    if not self._queue_closed_exception_types:
        self._queue_closed_exception_types = (errors.OutOfRangeError,)
    else:
        self._queue_closed_exception_types = tuple(self._queue_closed_exception_types)