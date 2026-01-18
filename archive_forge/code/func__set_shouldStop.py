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
def _set_shouldStop(self, value):
    if hasattr(self.decorated, 'shouldStop'):
        self.decorated.shouldStop = value
    else:
        self._shouldStop = value