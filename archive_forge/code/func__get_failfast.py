import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
def _get_failfast(self):
    return getattr(self.decorated, 'failfast', False)