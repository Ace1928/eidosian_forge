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
class DetailsContract(TagsContract):
    """Tests for the details API of TestResults."""

    def test_addExpectedFailure_details(self):
        result = self.makeResult()
        result.startTest(self)
        result.addExpectedFailure(self, details={})

    def test_addError_details(self):
        result = self.makeResult()
        result.startTest(self)
        result.addError(self, details={})

    def test_addFailure_details(self):
        result = self.makeResult()
        result.startTest(self)
        result.addFailure(self, details={})

    def test_addSkipped_details(self):
        result = self.makeResult()
        result.startTest(self)
        result.addSkip(self, details={})

    def test_addUnexpectedSuccess_details(self):
        result = self.makeResult()
        result.startTest(self)
        result.addUnexpectedSuccess(self, details={})

    def test_addSuccess_details(self):
        result = self.makeResult()
        result.startTest(self)
        result.addSuccess(self, details={})