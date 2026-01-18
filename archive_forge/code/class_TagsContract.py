import codecs
import datetime
import doctest
import io
from itertools import chain
from itertools import combinations
import os
import platform
from queue import Queue
import re
import shutil
import sys
import tempfile
import threading
from unittest import TestSuite
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.content_type import ContentType, UTF8_TEXT
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.tests.helpers import (
from testtools.testresult.doubles import (
from testtools.testresult.real import (
class TagsContract(Python27Contract):
    """Tests to ensure correct tagging behaviour.

    See the subunit docs for guidelines on how this is supposed to work.
    """

    def test_no_tags_by_default(self):
        result = self.makeResult()
        result.startTestRun()
        self.assertEqual(frozenset(), result.current_tags)

    def test_adding_tags(self):
        result = self.makeResult()
        result.startTestRun()
        result.tags({'foo'}, set())
        self.assertEqual({'foo'}, result.current_tags)

    def test_removing_tags(self):
        result = self.makeResult()
        result.startTestRun()
        result.tags({'foo'}, set())
        result.tags(set(), {'foo'})
        self.assertEqual(set(), result.current_tags)

    def test_startTestRun_resets_tags(self):
        result = self.makeResult()
        result.startTestRun()
        result.tags({'foo'}, set())
        result.startTestRun()
        self.assertEqual(set(), result.current_tags)

    def test_add_tags_within_test(self):
        result = self.makeResult()
        result.startTestRun()
        result.tags({'foo'}, set())
        result.startTest(self)
        result.tags({'bar'}, set())
        self.assertEqual({'foo', 'bar'}, result.current_tags)

    def test_tags_added_in_test_are_reverted(self):
        result = self.makeResult()
        result.startTestRun()
        result.tags({'foo'}, set())
        result.startTest(self)
        result.tags({'bar'}, set())
        result.addSuccess(self)
        result.stopTest(self)
        self.assertEqual({'foo'}, result.current_tags)

    def test_tags_removed_in_test(self):
        result = self.makeResult()
        result.startTestRun()
        result.tags({'foo'}, set())
        result.startTest(self)
        result.tags(set(), {'foo'})
        self.assertEqual(set(), result.current_tags)

    def test_tags_removed_in_test_are_restored(self):
        result = self.makeResult()
        result.startTestRun()
        result.tags({'foo'}, set())
        result.startTest(self)
        result.tags(set(), {'foo'})
        result.addSuccess(self)
        result.stopTest(self)
        self.assertEqual({'foo'}, result.current_tags)