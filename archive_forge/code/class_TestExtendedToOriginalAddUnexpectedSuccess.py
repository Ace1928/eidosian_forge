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
class TestExtendedToOriginalAddUnexpectedSuccess(TestExtendedToOriginalResultDecoratorBase):
    outcome = 'addUnexpectedSuccess'
    expected = 'addFailure'

    def test_outcome_Original_py26(self):
        self.make_26_result()
        getattr(self.converter, self.outcome)(self)
        [event] = self.result._events
        self.assertEqual((self.expected, self), event[:2])

    def test_outcome_Original_py27(self):
        self.make_27_result()
        self.check_outcome_nothing(self.outcome)

    def test_outcome_Original_pyextended(self):
        self.make_extended_result()
        self.check_outcome_nothing(self.outcome)

    def test_outcome_Extended_py26(self):
        self.make_26_result()
        getattr(self.converter, self.outcome)(self)
        [event] = self.result._events
        self.assertEqual((self.expected, self), event[:2])

    def test_outcome_Extended_py27(self):
        self.make_27_result()
        self.check_outcome_details_to_nothing(self.outcome)

    def test_outcome_Extended_pyextended(self):
        self.make_extended_result()
        self.check_outcome_details(self.outcome)