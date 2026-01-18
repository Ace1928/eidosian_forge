import datetime
import email.message
import math
from operator import methodcaller
import sys
import unittest
import warnings
from testtools.compat import _b
from testtools.content import (
from testtools.content_type import ContentType
from testtools.tags import TagContext
def _add_result_with_semaphore(self, method, test, *args, **kwargs):
    now = self._now()
    self.semaphore.acquire()
    try:
        self.result.time(self._test_start)
        self.result.startTest(test)
        self.result.time(now)
        if self._any_tags(self._global_tags):
            self.result.tags(*self._global_tags)
        if self._any_tags(self._test_tags):
            self.result.tags(*self._test_tags)
        self._test_tags = (set(), set())
        try:
            method(test, *args, **kwargs)
        finally:
            self.result.stopTest(test)
    finally:
        self.semaphore.release()
    self._test_start = None