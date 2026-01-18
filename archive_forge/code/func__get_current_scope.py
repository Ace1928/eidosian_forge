import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
def _get_current_scope(self):
    if self._test_tags:
        return self._test_tags
    return self._global_tags