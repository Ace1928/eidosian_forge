import os
import queue
import socket
import threading
import time
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
from tensorboard.summary.writer.record_writer import RecordWriter
class _AsyncWriter:
    """Writes bytes to a file."""

    def __init__(self, record_writer, max_queue_size=20, flush_secs=120):
        """Writes bytes to a file asynchronously. An instance of this class
        holds a queue to keep the incoming data temporarily. Data passed to the
        `write` function will be put to the queue and the function returns
        immediately. This class also maintains a thread to write data in the
        queue to disk. The first initialization parameter is an instance of
        `tensorboard.summary.record_writer` which computes the CRC checksum and
        then write the combined result to the disk. So we use an async approach
        to improve performance.

        Args:
            record_writer: A RecordWriter instance
            max_queue_size: Integer. Size of the queue for pending bytestrings.
            flush_secs: Number. How often, in seconds, to flush the
                pending bytestrings to disk.
        """
        self._writer = record_writer
        self._closed = False
        self._byte_queue = queue.Queue(max_queue_size)
        self._worker = _AsyncWriterThread(self._byte_queue, self._writer, flush_secs)
        self._lock = threading.Lock()
        self._worker.start()

    def write(self, bytestring):
        """Enqueue the given bytes to be written asychronously."""
        with self._lock:
            self._check_worker_status()
            if self._closed:
                raise IOError('Writer is closed')
            self._byte_queue.put(bytestring)
            self._check_worker_status()

    def flush(self):
        """Write all the enqueued bytestring before this flush call to disk.

        Block until all the above bytestring are written.
        """
        with self._lock:
            self._check_worker_status()
            if self._closed:
                raise IOError('Writer is closed')
            self._byte_queue.join()
            self._writer.flush()
            self._check_worker_status()

    def close(self):
        """Closes the underlying writer, flushing any pending writes first."""
        if not self._closed:
            with self._lock:
                if not self._closed:
                    self._closed = True
                    self._worker.stop()
                    self._writer.flush()
                    self._writer.close()

    def _check_worker_status(self):
        """Makes sure the worker thread is still running and raises exception
        thrown in the worker thread otherwise.
        """
        exception = self._worker.exception
        if exception is not None:
            raise exception