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
class TestStreamResultContract:

    def _make_result(self):
        raise NotImplementedError(self._make_result)

    def test_startTestRun(self):
        result = self._make_result()
        result.startTestRun()
        result.stopTestRun()

    def test_files(self):
        result = self._make_result()
        result.startTestRun()
        self.addCleanup(result.stopTestRun)
        now = datetime.datetime.now(utc)
        inputs = list(dict(eof=True, mime_type='text/plain', route_code='1234', test_id='foo', timestamp=now).items())
        param_dicts = self._power_set(inputs)
        for kwargs in param_dicts:
            result.status(file_name='foo', file_bytes=_b(''), **kwargs)
            result.status(file_name='foo', file_bytes=_b('bar'), **kwargs)

    def test_test_status(self):
        result = self._make_result()
        result.startTestRun()
        self.addCleanup(result.stopTestRun)
        now = datetime.datetime.now(utc)
        args = [['foo', s] for s in ['exists', 'inprogress', 'xfail', 'uxsuccess', 'success', 'fail', 'skip']]
        inputs = list(dict(runnable=False, test_tags={'quux'}, route_code='1234', timestamp=now).items())
        param_dicts = self._power_set(inputs)
        for kwargs in param_dicts:
            for arg in args:
                result.status(test_id=arg[0], test_status=arg[1], **kwargs)

    def _power_set(self, iterable):
        """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
        s = list(iterable)
        param_dicts = []
        combos = (combinations(s, r) for r in range(len(s) + 1))
        for ss in chain.from_iterable(combos):
            param_dicts.append(dict(ss))
        return param_dicts