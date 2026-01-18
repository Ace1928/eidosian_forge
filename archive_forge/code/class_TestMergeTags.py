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
class TestMergeTags(TestCase):

    def test_merge_unseen_gone_tag(self):
        current_tags = ({'present'}, {'missing'})
        changing_tags = (set(), {'going'})
        expected = ({'present'}, {'missing', 'going'})
        self.assertEqual(expected, _merge_tags(current_tags, changing_tags))

    def test_merge_incoming_gone_tag_with_current_new_tag(self):
        current_tags = ({'present', 'going'}, {'missing'})
        changing_tags = (set(), {'going'})
        expected = ({'present'}, {'missing', 'going'})
        self.assertEqual(expected, _merge_tags(current_tags, changing_tags))

    def test_merge_unseen_new_tag(self):
        current_tags = ({'present'}, {'missing'})
        changing_tags = ({'coming'}, set())
        expected = ({'coming', 'present'}, {'missing'})
        self.assertEqual(expected, _merge_tags(current_tags, changing_tags))

    def test_merge_incoming_new_tag_with_current_gone_tag(self):
        current_tags = ({'present'}, {'coming', 'missing'})
        changing_tags = ({'coming'}, set())
        expected = ({'coming', 'present'}, {'missing'})
        self.assertEqual(expected, _merge_tags(current_tags, changing_tags))