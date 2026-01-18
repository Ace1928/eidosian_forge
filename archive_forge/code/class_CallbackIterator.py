import urllib
from oslo_log import log as logging
from oslo_utils import timeutils
from glance.common import exception
from glance.i18n import _, _LE
class CallbackIterator(object):
    """A proxy iterator that calls a callback function periodically

    This is used to wrap a reading file object and proxy its chunks
    through to another caller. Periodically, the callback function
    will be called with information about the data processed so far,
    allowing for status updating or cancel flag checking. The function
    can be called every time we process a chunk, or only after we have
    processed a certain amount of data since the last call.

    :param source: A source iterator whose content will be proxied
                   through this object.
    :param callback: A function to be called periodically while iterating.
                     The signature should be fn(chunk_bytes, total_bytes),
                     where chunk is the number of bytes since the last
                     call of the callback, and total_bytes is the total amount
                     copied thus far.
    :param min_interval: Limit the calls to callback to only when this many
                         seconds have elapsed since the last callback (a
                         close() or final iteration may fire the callback in
                         less time to ensure completion).
    """

    def __init__(self, source, callback, min_interval=None):
        self._source = source
        self._callback = callback
        self._min_interval = min_interval
        self._chunk_bytes = 0
        self._total_bytes = 0
        self._timer = None

    @property
    def callback_due(self):
        """Indicates if a callback should be made.

        If no time-based limit is set, this will always be True.
        If a limit is set, then this returns True exactly once,
        resetting the timer when it does.
        """
        if not self._min_interval:
            return True
        if not self._timer:
            self._timer = timeutils.StopWatch(self._min_interval)
            self._timer.start()
        if self._timer.expired():
            self._timer.restart()
            return True
        else:
            return False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._source)
        except StopIteration:
            self._call_callback(b'', is_last=True)
            raise
        self._call_callback(chunk)
        return chunk

    def close(self):
        self._call_callback(b'', is_last=True)
        if hasattr(self._source, 'close'):
            return self._source.close()

    def _call_callback(self, chunk, is_last=False):
        self._total_bytes += len(chunk)
        self._chunk_bytes += len(chunk)
        if not self._chunk_bytes:
            return
        if is_last or self.callback_due:
            self._callback(self._chunk_bytes, self._total_bytes)
            self._chunk_bytes = 0

    def read(self, size=None):
        chunk = self._source.read(size)
        self._call_callback(chunk)
        return chunk