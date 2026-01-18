from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import threading
import tensorflow as tf
def _run(self, sess, enqueue_op, feed_fn, coord=None):
    """Execute the enqueue op in a loop, close the queue in case of error.

    Args:
      sess: A `Session`.
      enqueue_op: The `Operation` to run.
      feed_fn: the feed function to pass to `sess.run`.
      coord: Optional `Coordinator` object for reporting errors and checking for
        stop conditions.
    """
    if coord:
        coord.register_thread(threading.current_thread())
    decremented = False
    try:
        while True:
            if coord and coord.should_stop():
                break
            try:
                feed_dict = None if feed_fn is None else feed_fn()
                sess.run(enqueue_op, feed_dict=feed_dict)
            except (tf.errors.OutOfRangeError, tf.errors.CancelledError):
                with self._lock:
                    self._runs_per_session[sess] -= 1
                    decremented = True
                    if self._runs_per_session[sess] == 0:
                        try:
                            sess.run(self._close_op)
                        except Exception as e:
                            tf.compat.v1.logging.vlog(1, 'Ignored exception: %s', str(e))
                    return
    except Exception as e:
        if coord:
            coord.request_stop(e)
        else:
            tf.compat.v1.logging.error('Exception in QueueRunner: %s', str(e))
            with self._lock:
                self._exceptions_raised.append(e)
            raise
    finally:
        if not decremented:
            with self._lock:
                self._runs_per_session[sess] -= 1