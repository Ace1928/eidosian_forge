import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
def _flush_current_scope(self, tag_receiver):
    new_tags, gone_tags = self._get_current_scope()
    if new_tags or gone_tags:
        tag_receiver.tags(new_tags, gone_tags)
    if self._test_tags:
        self._test_tags = (set(), set())
    else:
        self._global_tags = (set(), set())