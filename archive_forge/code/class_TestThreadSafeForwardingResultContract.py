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
class TestThreadSafeForwardingResultContract(TestCase, StartTestRunContract):
    run_test_with = FullStackRunTest

    def makeResult(self):
        result_semaphore = threading.Semaphore(1)
        target = TestResult()
        return ThreadsafeForwardingResult(target, result_semaphore)