import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
def _on_test(self, test, status, start_time, stop_time, tags, details):
    self._write_row([test.id(), status, start_time, stop_time])