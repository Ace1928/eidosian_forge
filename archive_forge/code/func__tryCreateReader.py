import threading
from tensorboard import errors
def _tryCreateReader(self):
    """Try creating reader for tfdbg2 data in the logdir.

        If the reader has already been created, a new one will not be created and
        this function is a no-op.

        If a reader has not been created, create it and start periodic calls to
        `update()` on a separate thread.
        """
    if self._reader:
        return
    with self._reader_lock:
        if not self._reader:
            try:
                from tensorflow.python.debug.lib import debug_events_reader
                from tensorflow.python.debug.lib import debug_events_monitors
            except ImportError:
                return
            try:
                self._reader = debug_events_reader.DebugDataReader(self._logdir)
            except AttributeError:
                return
            except ValueError:
                return
            self._monitors = [debug_events_monitors.InfNanMonitor(self._reader, limit=DEFAULT_PER_TYPE_ALERT_LIMIT)]
            self._reload_needed_event, _ = run_repeatedly_in_background(self._reader.update, DEFAULT_RELOAD_INTERVAL_SEC)