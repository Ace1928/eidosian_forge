import select
from pyudev._util import eintr_retry_call
def _parse_events(self, events):
    """Parse ``events``.

        ``events`` is a list of events as returned by
        :meth:`select.poll.poll()`.

        Yield all parsed events.

        """
    for fd, event_mask in events:
        if self._has_event(event_mask, select.POLLNVAL):
            raise IOError(f'File descriptor not open: {repr(fd)}')
        if self._has_event(event_mask, select.POLLERR):
            raise IOError(f'Error while polling fd: {repr(fd)}')
        if self._has_event(event_mask, select.POLLIN):
            yield (fd, 'r')
        if self._has_event(event_mask, select.POLLOUT):
            yield (fd, 'w')
        if self._has_event(event_mask, select.POLLHUP):
            yield (fd, 'h')