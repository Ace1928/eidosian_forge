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
class CopyStreamResult(StreamResult):
    """Copies all event it receives to multiple results.

    This provides an easy facility for combining multiple StreamResults.

    For TestResult the equivalent class was ``MultiTestResult``.
    """

    def __init__(self, targets):
        super().__init__()
        self.targets = targets

    def startTestRun(self):
        super().startTestRun()
        _strict_map(methodcaller('startTestRun'), self.targets)

    def stopTestRun(self):
        super().stopTestRun()
        _strict_map(methodcaller('stopTestRun'), self.targets)

    def status(self, *args, **kwargs):
        super().status(*args, **kwargs)
        _strict_map(methodcaller('status', *args, **kwargs), self.targets)