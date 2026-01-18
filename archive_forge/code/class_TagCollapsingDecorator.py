import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
class TagCollapsingDecorator(HookedTestResultDecorator, TagsMixin):
    """Collapses many 'tags' calls into one where possible."""

    def __init__(self, result):
        super().__init__(result)
        self._clear_tags()

    def _before_event(self):
        self._flush_current_scope(self.decorated)

    def tags(self, new_tags, gone_tags):
        TagsMixin.tags(self, new_tags, gone_tags)