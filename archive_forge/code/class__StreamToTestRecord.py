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
class _StreamToTestRecord(StreamResult):
    """A specialised StreamResult that emits a callback as tests complete.

    Top level file attachments are simply discarded. Hung tests are detected
    by stopTestRun and notified there and then.

    The callback is passed a ``_TestRecord`` object.

    Only the most recent tags observed in the stream are reported.
    """

    def __init__(self, on_test):
        """Create a _StreamToTestRecord calling on_test on test completions.

        :param on_test: A callback that accepts one parameter:
            a ``_TestRecord`` object describing a test.
        """
        super().__init__()
        self.on_test = on_test

    def startTestRun(self):
        super().startTestRun()
        self._inprogress = {}

    def status(self, test_id=None, test_status=None, test_tags=None, runnable=True, file_name=None, file_bytes=None, eof=False, mime_type=None, route_code=None, timestamp=None):
        super().status(test_id, test_status, test_tags=test_tags, runnable=runnable, file_name=file_name, file_bytes=file_bytes, eof=eof, mime_type=mime_type, route_code=route_code, timestamp=timestamp)
        key = self._ensure_key(test_id, route_code, timestamp)
        if not key:
            return
        self._inprogress[key] = self._update_case(self._inprogress[key], test_status, test_tags, file_name, file_bytes, mime_type, timestamp)
        if test_status not in INTERIM_STATES:
            self.on_test(self._inprogress.pop(key))

    def _update_case(self, case, test_status=None, test_tags=None, file_name=None, file_bytes=None, mime_type=None, timestamp=None):
        if test_status is not None:
            case = case.set(status=test_status)
        case = case.got_timestamp(timestamp)
        if file_name is not None and file_bytes:
            case = case.got_file(file_name, file_bytes, mime_type)
        if test_tags is not None:
            case = case.set('tags', test_tags)
        return case

    def stopTestRun(self):
        super().stopTestRun()
        while self._inprogress:
            case = self._inprogress.popitem()[1]
            self.on_test(case.got_timestamp(None))

    def _ensure_key(self, test_id, route_code, timestamp):
        if test_id is None:
            return
        key = (test_id, route_code)
        if key not in self._inprogress:
            self._inprogress[key] = _TestRecord.create(test_id, timestamp)
        return key