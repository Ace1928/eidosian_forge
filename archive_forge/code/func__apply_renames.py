import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
def _apply_renames(self, test):
    if self._rename_fn is None:
        return test
    new_id = self._rename_fn(test.id())
    setattr(test, 'id', lambda: new_id)
    return test