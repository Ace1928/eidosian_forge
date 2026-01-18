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
class TestExtendedToOriginalAddSkip(TestExtendedToOriginalResultDecoratorBase):
    outcome = 'addSkip'

    def test_outcome_Original_py26(self):
        self.make_26_result()
        self.check_outcome_string_nothing(self.outcome, 'addSuccess')

    def test_outcome_Original_py27(self):
        self.make_27_result()
        self.check_outcome_string(self.outcome)

    def test_outcome_Original_pyextended(self):
        self.make_extended_result()
        self.check_outcome_string(self.outcome)

    def test_outcome_Extended_py26(self):
        self.make_26_result()
        self.check_outcome_string_nothing(self.outcome, 'addSuccess')

    def test_outcome_Extended_py27_no_reason(self):
        self.make_27_result()
        self.check_outcome_details_to_string(self.outcome)

    def test_outcome_Extended_py27_reason(self):
        self.make_27_result()
        self.check_outcome_details_to_arg(self.outcome, 'foo', {'reason': Content(UTF8_TEXT, lambda: [_b('foo')])})

    def test_outcome_Extended_pyextended(self):
        self.make_extended_result()
        self.check_outcome_details(self.outcome)

    def test_outcome__no_details(self):
        self.make_extended_result()
        self.assertThat(lambda: getattr(self.converter, self.outcome)(self), Raises(MatchesException(ValueError)))