import ast
import io
import sys
import traceback
import testtools
from testtools.compat import (
from testtools.matchers import (
class _FakeOutputStream:
    """A simple file-like object for testing"""

    def __init__(self):
        self.writelog = []

    def write(self, obj):
        self.writelog.append(obj)