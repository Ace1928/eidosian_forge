import collections
import os.path
import sys
import threading
import time
from tensorflow.python.client import _pywrap_events_writer
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
class EventFileWriter:
    """Writes `Event` protocol buffers to an event file.

  The `EventFileWriter` class creates an event file in the specified directory,
  and asynchronously writes Event protocol buffers to the file. The Event file
  is encoded using the tfrecord format, which is similar to RecordIO.

  This class is not thread-safe.
  """

    def __init__(self, logdir, max_queue=10, flush_secs=120, filename_suffix=None):
        """Creates a `EventFileWriter` and an event file to write to.

    On construction the summary writer creates a new event file in `logdir`.
    This event file will contain `Event` protocol buffers, which are written to
    disk via the add_event method.

    The other arguments to the constructor control the asynchronous writes to
    the event file:

    *  `flush_secs`: How often, in seconds, to flush the added summaries
       and events to disk.
    *  `max_queue`: Maximum number of summaries or events pending to be
       written to disk before one of the 'add' calls block.

    Args:
      logdir: A string. Directory where event file will be written.
      max_queue: Integer. Size of the queue for pending events and summaries.
      flush_secs: Number. How often, in seconds, to flush the
        pending events and summaries to disk.
      filename_suffix: A string. Every event file's name is suffixed with
        `filename_suffix`.
    """
        self._logdir = str(logdir)
        gfile.MakeDirs(self._logdir)
        self._max_queue = max_queue
        self._flush_secs = flush_secs
        self._flush_complete = threading.Event()
        self._flush_sentinel = object()
        self._close_sentinel = object()
        self._ev_writer = _pywrap_events_writer.EventsWriter(compat.as_bytes(os.path.join(self._logdir, 'events')))
        if filename_suffix:
            self._ev_writer.InitWithSuffix(compat.as_bytes(filename_suffix))
        self._initialize()
        self._closed = False

    def _initialize(self):
        """Initializes or re-initializes the queue and writer thread.

    The EventsWriter itself does not need to be re-initialized explicitly,
    because it will auto-initialize itself if used after being closed.
    """
        self._event_queue = CloseableQueue(self._max_queue)
        self._worker = _EventLoggerThread(self._event_queue, self._ev_writer, self._flush_secs, self._flush_complete, self._flush_sentinel, self._close_sentinel)
        self._worker.start()

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self._logdir

    def reopen(self):
        """Reopens the EventFileWriter.

    Can be called after `close()` to add more events in the same directory.
    The events will go into a new events file.

    Does nothing if the EventFileWriter was not closed.
    """
        if self._closed:
            self._initialize()
            self._closed = False

    def add_event(self, event):
        """Adds an event to the event file.

    Args:
      event: An `Event` protocol buffer.
    """
        if not self._closed:
            self._try_put(event)

    def _try_put(self, item):
        """Attempts to enqueue an item to the event queue.

    If the queue is closed, this will close the EventFileWriter and reraise the
    exception that caused the queue closure, if one exists.

    Args:
      item: the item to enqueue
    """
        try:
            self._event_queue.put(item)
        except QueueClosedError:
            self._internal_close()
            if self._worker.failure_exc_info:
                _, exception, _ = self._worker.failure_exc_info
                raise exception from None

    def flush(self):
        """Flushes the event file to disk.

    Call this method to make sure that all pending events have been written to
    disk.
    """
        if not self._closed:
            self._flush_complete.clear()
            self._try_put(self._flush_sentinel)
            self._flush_complete.wait()
            if self._worker.failure_exc_info:
                self._internal_close()
                _, exception, _ = self._worker.failure_exc_info
                raise exception

    def close(self):
        """Flushes the event file to disk and close the file.

    Call this method when you do not need the summary writer anymore.
    """
        if not self._closed:
            self.flush()
            self._try_put(self._close_sentinel)
            self._internal_close()

    def _internal_close(self):
        self._closed = True
        self._worker.join()
        self._ev_writer.Close()