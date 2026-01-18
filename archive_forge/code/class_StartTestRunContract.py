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
class StartTestRunContract(FallbackContract):
    """Defines the contract for testtools policy choices.

    That is things which are not simply extensions to unittest but choices we
    have made differently.
    """

    def test_startTestRun_resets_unexpected_success(self):
        result = self.makeResult()
        result.startTest(self)
        result.addUnexpectedSuccess(self)
        result.stopTest(self)
        result.startTestRun()
        self.assertTrue(result.wasSuccessful())

    def test_startTestRun_resets_failure(self):
        result = self.makeResult()
        result.startTest(self)
        result.addFailure(self, an_exc_info)
        result.stopTest(self)
        result.startTestRun()
        self.assertTrue(result.wasSuccessful())

    def test_startTestRun_resets_errors(self):
        result = self.makeResult()
        result.startTest(self)
        result.addError(self, an_exc_info)
        result.stopTest(self)
        result.startTestRun()
        self.assertTrue(result.wasSuccessful())