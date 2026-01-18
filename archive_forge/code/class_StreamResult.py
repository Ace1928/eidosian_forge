from collections import namedtuple
from testtools.tags import TagContext
class StreamResult(LoggingBase):
    """A StreamResult implementation for testing.

    All events are logged to _events.
    """

    def startTestRun(self):
        self._events.append(('startTestRun',))

    def stopTestRun(self):
        self._events.append(('stopTestRun',))

    def status(self, test_id=None, test_status=None, test_tags=None, runnable=True, file_name=None, file_bytes=None, eof=False, mime_type=None, route_code=None, timestamp=None):
        self._events.append(_StatusEvent('status', test_id, test_status, test_tags, runnable, file_name, file_bytes, eof, mime_type, route_code, timestamp))