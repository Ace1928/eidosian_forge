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
class TestStreamFailFast(TestCase):

    def test_inprogress(self):
        result = StreamFailFast(self.fail)
        result.status('foo', 'inprogress')

    def test_exists(self):
        result = StreamFailFast(self.fail)
        result.status('foo', 'exists')

    def test_xfail(self):
        result = StreamFailFast(self.fail)
        result.status('foo', 'xfail')

    def test_uxsuccess(self):
        calls = []

        def hook():
            calls.append('called')
        result = StreamFailFast(hook)
        result.status('foo', 'uxsuccess')
        result.status('foo', 'uxsuccess')
        self.assertEqual(['called', 'called'], calls)

    def test_success(self):
        result = StreamFailFast(self.fail)
        result.status('foo', 'success')

    def test_fail(self):
        calls = []

        def hook():
            calls.append('called')
        result = StreamFailFast(hook)
        result.status('foo', 'fail')
        result.status('foo', 'fail')
        self.assertEqual(['called', 'called'], calls)

    def test_skip(self):
        result = StreamFailFast(self.fail)
        result.status('foo', 'skip')