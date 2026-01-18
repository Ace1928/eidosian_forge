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
class _CoordinatedClosureQueue(object):
    """Manage a queue of closures, inflight count and errors from execution.

  This class is thread-safe.
  """

    def __init__(self):
        self.inflight_closure_count = 0
        self._queue_lock = threading.Lock()
        self._stop_waiting_condition = threading.Condition(self._queue_lock)
        self._closures_queued_condition = threading.Condition(self._queue_lock)
        self._should_process_closures = True
        self._queue_free_slot_condition = threading.Condition(self._queue_lock)
        self._no_inflight_closure_condition = threading.Condition(self._queue_lock)
        self._cancellation_mgr = cancellation.CancellationManager()
        if _CLOSURE_QUEUE_MAX_SIZE <= 0:
            logging.warning('In a `ClusterCoordinator`, creating an infinite closure queue can consume a significant amount of memory and even lead to OOM.')
        self._queue = queue.Queue(maxsize=_CLOSURE_QUEUE_MAX_SIZE)
        metric_utils.monitor_int('queued_closures', self._queue.qsize())
        self._tagged_queue = collections.defaultdict(queue.Queue)
        self._error = None
        self._put_wait_lock = threading.Lock()
        self._watchdog = watchdog.WatchDog(on_triggered=self._on_watchdog_timeout)

    def _on_watchdog_timeout(self):
        logging.info('inflight_closure_count is %d', self._inflight_closure_count)
        logging.info('current error is %s:%r', self._error, self._error)

    @property
    def inflight_closure_count(self):
        return self._inflight_closure_count

    @inflight_closure_count.setter
    def inflight_closure_count(self, value):
        self._inflight_closure_count = value
        metric_utils.monitor_int('inflight_closures', self._inflight_closure_count)

    def stop(self):
        with self._queue_lock:
            self._should_process_closures = False
            self._cancellation_mgr.start_cancel()
            self._closures_queued_condition.notify_all()
        self._watchdog.stop()

    def _cancel_all_closures(self):
        """Clears the queue and sets remaining closures cancelled error.

    This method expects self._queue_lock to be held prior to entry.
    """
        self._cancellation_mgr.start_cancel()
        logging.info('Canceling all closures: waiting for inflight closures to finish')
        while self._inflight_closure_count > 0:
            self._no_inflight_closure_condition.wait()
        logging.info('Canceling all closures: canceling remaining closures on the queue')
        while True:
            try:
                closure = self._queue.get(block=False)
                metric_utils.monitor_int('queued_closures', self._queue.qsize())
                self._queue_free_slot_condition.notify()
                closure.mark_cancelled()
            except queue.Empty:
                break
        self._cancellation_mgr = cancellation.CancellationManager()

    def _raise_if_error(self):
        """Raises the error if one exists.

    If an error exists, cancel the closures in queue, raises it, and clear
    the error.

    This method expects self._queue_lock to be held prior to entry.
    """
        if self._error:
            logging.error('Start cancelling closures due to error %r: %s', self._error, self._error)
            self._cancel_all_closures()
            try:
                raise self._error
            finally:
                self._error = None

    def put(self, closure, tag=None):
        """Put a closure into the queue for later execution.

    If `mark_failed` was called before `put`, the error from the first
    invocation of `mark_failed` will be raised.

    Args:
      closure: The `Closure` to put into the queue.
      tag: if not None, put into a queue with the given tag.
    """
        closure.tag = tag
        if tag is not None:
            with self._queue_lock:
                self._tagged_queue[tag].put(closure, block=False)
                self._closures_queued_condition.notify_all()
        else:
            with self._put_wait_lock, self._queue_lock:
                self._queue_free_slot_condition.wait_for(lambda: not self._queue.full())
                self._queue.put(closure, block=False)
                metric_utils.monitor_int('queued_closures', self._queue.qsize())
                self._raise_if_error()
                self._closures_queued_condition.notify()

    def get(self, timeout=None, tag=None):
        """Return a closure from the queue to be executed.

    It will try to fetch an item from the queue with the given tag. If this
    queue is empty, it will then check the global queue.

    Args:
      timeout: timeout when waiting for a closure to be put.
      tag: optional tag to specify which queue to query first before querying
        the global queue.

    Returns:
      a closure or None after timeout.
    """
        with self._queue_lock:
            while self._should_process_closures and self._queue.empty() and (tag is None or self._tagged_queue[tag].empty()):
                if not self._closures_queued_condition.wait(timeout=timeout):
                    return None
            if not self._should_process_closures:
                return None
            if tag is not None and (not self._tagged_queue[tag].empty()):
                closure = self._tagged_queue[tag].get(block=False)
                return closure
            closure = self._queue.get(block=False)
            metric_utils.monitor_int('queued_closures', self._queue.qsize())
            assert closure.tag is None
            assert tag is None or self._tagged_queue[tag].empty()
            self._queue_free_slot_condition.notify()
            self.inflight_closure_count += 1
            return closure

    def mark_finished(self):
        """Let the queue know that a closure has been successfully executed."""
        with self._queue_lock:
            if self._inflight_closure_count < 1:
                raise AssertionError('There is no inflight closures to mark_finished.')
            self.inflight_closure_count -= 1
            if self._inflight_closure_count == 0:
                self._no_inflight_closure_condition.notify_all()
            if self._queue.empty() and self._inflight_closure_count == 0:
                self._stop_waiting_condition.notify_all()
            self._watchdog.report_closure_done()

    def put_back(self, closure):
        """Put the closure back into the queue as it was not properly executed."""
        assert closure.tag is None
        with self._queue_lock:
            if self._inflight_closure_count < 1:
                raise AssertionError('There is no inflight closures to put_back.')
            if self._error:
                closure.mark_cancelled()
            else:
                self._queue_free_slot_condition.wait_for(lambda: not self._queue.full())
                self._queue.put(closure, block=False)
                metric_utils.monitor_int('queued_closures', self._queue.qsize())
                self._closures_queued_condition.notify()
            self.inflight_closure_count -= 1
            if self._inflight_closure_count == 0:
                self._no_inflight_closure_condition.notify_all()

    def wait(self, timeout=None):
        """Wait for all closures to be finished before returning.

    If `mark_failed` was called before or during `wait`, the error from the
    first invocation of `mark_failed` will be raised.

    Args:
      timeout: A float specifying a timeout for the wait in seconds.

    Returns:
      True unless the given timeout expired, in which case it returns False.
    """
        with self._put_wait_lock, self._queue_lock:
            logging.info('Waiting for all global closures to be finished.')
            while not self._error and (not self._queue.empty() or self._inflight_closure_count > 0):
                if not self._stop_waiting_condition.wait(timeout=timeout):
                    return False
            self._raise_if_error()
            return True

    def mark_failed(self, e):
        """Sets error and unblocks any wait() call."""
        with self._queue_lock:
            if self._inflight_closure_count < 1:
                raise AssertionError('There is no inflight closures to mark_failed.')
            if self._error is None:
                self._error = e
            self.inflight_closure_count -= 1
            if self._inflight_closure_count == 0:
                self._no_inflight_closure_condition.notify_all()
            self._stop_waiting_condition.notify_all()

    def done(self):
        """Returns true if the queue is empty and there is no inflight closure.

    If `mark_failed` was called before `done`, the error from the first
    invocation of `mark_failed` will be raised.
    """
        with self._queue_lock:
            self._raise_if_error()
            return self._queue.empty() and self._inflight_closure_count == 0

    def clear_tag_unlocked(self, tag):
        self._tagged_queue[tag] = queue.Queue()