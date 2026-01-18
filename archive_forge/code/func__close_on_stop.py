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
def _close_on_stop(self, sess, cancel_op, coord):
    """Close the queue when the Coordinator requests stop.

    Args:
      sess: A Session.
      cancel_op: The Operation to run.
      coord: Coordinator.
    """
    coord.wait_for_stop()
    try:
        sess.run(cancel_op)
    except Exception as e:
        logging.vlog(1, 'Ignored exception: %s', str(e))