import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
class TagsMixin:

    def __init__(self):
        self._clear_tags()

    def _clear_tags(self):
        self._global_tags = (set(), set())
        self._test_tags = None

    def _get_active_tags(self):
        global_new, global_gone = self._global_tags
        if self._test_tags is None:
            return set(global_new)
        test_new, test_gone = self._test_tags
        return global_new.difference(test_gone).union(test_new)

    def _get_current_scope(self):
        if self._test_tags:
            return self._test_tags
        return self._global_tags

    def _flush_current_scope(self, tag_receiver):
        new_tags, gone_tags = self._get_current_scope()
        if new_tags or gone_tags:
            tag_receiver.tags(new_tags, gone_tags)
        if self._test_tags:
            self._test_tags = (set(), set())
        else:
            self._global_tags = (set(), set())

    def startTestRun(self):
        self._clear_tags()

    def startTest(self, test):
        self._test_tags = (set(), set())

    def stopTest(self, test):
        self._test_tags = None

    def tags(self, new_tags, gone_tags):
        """Handle tag instructions.

        Adds and removes tags as appropriate. If a test is currently running,
        tags are not affected for subsequent tests.

        :param new_tags: Tags to add,
        :param gone_tags: Tags to remove.
        """
        current_new_tags, current_gone_tags = self._get_current_scope()
        current_new_tags.update(new_tags)
        current_new_tags.difference_update(gone_tags)
        current_gone_tags.update(gone_tags)
        current_gone_tags.difference_update(new_tags)