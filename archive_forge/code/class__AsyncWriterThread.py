import os
import queue
import socket
import threading
import time
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
from tensorboard.summary.writer.record_writer import RecordWriter
class _AsyncWriterThread(threading.Thread):
    """Thread that processes asynchronous writes for _AsyncWriter."""

    def __init__(self, queue, record_writer, flush_secs):
        """Creates an _AsyncWriterThread.

        Args:
          queue: A Queue from which to dequeue data.
          record_writer: An instance of record_writer writer.
          flush_secs: How often, in seconds, to flush the
            pending file to disk.
        """
        threading.Thread.__init__(self)
        self.daemon = True
        self.exception = None
        self._queue = queue
        self._record_writer = record_writer
        self._flush_secs = flush_secs
        self._next_flush_time = 0
        self._has_pending_data = False
        self._shutdown_signal = object()

    def stop(self):
        self._queue.put(self._shutdown_signal)
        self.join()

    def run(self):
        try:
            self._run()
        except Exception as ex:
            self.exception = ex
            try:
                while True:
                    self._queue.get(False)
                    self._queue.task_done()
            except queue.Empty:
                pass
            raise

    def _run(self):
        while True:
            now = time.time()
            queue_wait_duration = self._next_flush_time - now
            data = None
            try:
                if queue_wait_duration > 0:
                    data = self._queue.get(True, queue_wait_duration)
                else:
                    data = self._queue.get(False)
                if data is self._shutdown_signal:
                    return
                self._record_writer.write(data)
                self._has_pending_data = True
            except queue.Empty:
                pass
            finally:
                if data:
                    self._queue.task_done()
            now = time.time()
            if now > self._next_flush_time:
                if self._has_pending_data:
                    self._record_writer.flush()
                    self._has_pending_data = False
                self._next_flush_time = now + self._flush_secs