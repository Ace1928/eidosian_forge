import datetime
import email.message
import math
from operator import methodcaller
import sys
import unittest
import warnings
from testtools.compat import _b
from testtools.content import (
from testtools.content_type import ContentType
from testtools.tags import TagContext
class StreamToQueue(StreamResult):
    """A StreamResult which enqueues events as a dict to a queue.Queue.

    Events have their route code updated to include the route code
    StreamToQueue was constructed with before they are submitted. If the event
    route code is None, it is replaced with the StreamToQueue route code,
    otherwise it is prefixed with the supplied code + a hyphen.

    startTestRun and stopTestRun are forwarded to the queue. Implementors that
    dequeue events back into StreamResult calls should take care not to call
    startTestRun / stopTestRun on other StreamResult objects multiple times
    (e.g. by filtering startTestRun and stopTestRun).

    ``StreamToQueue`` is typically used by
    ``ConcurrentStreamTestSuite``, which creates one ``StreamToQueue``
    per thread, forwards status events to the the StreamResult that
    ``ConcurrentStreamTestSuite.run()`` was called with, and uses the
    stopTestRun event to trigger calling join() on the each thread.

    Unlike ThreadsafeForwardingResult which this supercedes, no buffering takes
    place - any event supplied to a StreamToQueue will be inserted into the
    queue immediately.

    Events are forwarded as a dict with a key ``event`` which is one of
    ``startTestRun``, ``stopTestRun`` or ``status``. When ``event`` is
    ``status`` the dict also has keys matching the keyword arguments
    of ``StreamResult.status``, otherwise it has one other key ``result`` which
    is the result that invoked ``startTestRun``.
    """

    def __init__(self, queue, routing_code):
        """Create a StreamToQueue forwarding to target.

        :param queue: A ``queue.Queue`` to receive events.
        :param routing_code: The routing code to apply to messages.
        """
        super().__init__()
        self.queue = queue
        self.routing_code = routing_code

    def startTestRun(self):
        self.queue.put(dict(event='startTestRun', result=self))

    def status(self, test_id=None, test_status=None, test_tags=None, runnable=True, file_name=None, file_bytes=None, eof=False, mime_type=None, route_code=None, timestamp=None):
        self.queue.put(dict(event='status', test_id=test_id, test_status=test_status, test_tags=test_tags, runnable=runnable, file_name=file_name, file_bytes=file_bytes, eof=eof, mime_type=mime_type, route_code=self.route_code(route_code), timestamp=timestamp))

    def stopTestRun(self):
        self.queue.put(dict(event='stopTestRun', result=self))

    def route_code(self, route_code):
        """Adjust route_code on the way through."""
        if route_code is None:
            return self.routing_code
        return self.routing_code + '/' + route_code