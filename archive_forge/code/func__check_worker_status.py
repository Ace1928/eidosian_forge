import os
import queue
import socket
import threading
import time
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
from tensorboard.summary.writer.record_writer import RecordWriter
def _check_worker_status(self):
    """Makes sure the worker thread is still running and raises exception
        thrown in the worker thread otherwise.
        """
    exception = self._worker.exception
    if exception is not None:
        raise exception