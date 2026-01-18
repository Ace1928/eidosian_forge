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
class StreamToDict(StreamResult):
    """A specialised StreamResult that emits a callback as tests complete.

    Top level file attachments are simply discarded. Hung tests are detected
    by stopTestRun and notified there and then.

    The callback is passed a dict with the following keys:

      * id: the test id.
      * tags: The tags for the test. A set of unicode strings.
      * details: A dict of file attachments - ``testtools.content.Content``
        objects.
      * status: One of the StreamResult status codes (including inprogress) or
        'unknown' (used if only file events for a test were received...)
      * timestamps: A pair of timestamps - the first one received with this
        test id, and the one in the event that triggered the notification.
        Hung tests have a None for the second end event. Timestamps are not
        compared - their ordering is purely order received in the stream.

    Only the most recent tags observed in the stream are reported.
    """

    def __init__(self, on_test):
        """Create a _StreamToTestRecord calling on_test on test completions.

        :param on_test: A callback that accepts one parameter:
            a dictionary describing a test.
        """
        super().__init__()
        self._hook = _StreamToTestRecord(self._handle_test)
        self.on_test = on_test

    def _handle_test(self, test_record):
        self.on_test(test_record.to_dict())

    def startTestRun(self):
        super().startTestRun()
        self._hook.startTestRun()

    def status(self, *args, **kwargs):
        super().status(*args, **kwargs)
        self._hook.status(*args, **kwargs)

    def stopTestRun(self):
        super().stopTestRun()
        self._hook.stopTestRun()