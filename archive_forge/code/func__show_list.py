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
def _show_list(self, label, error_list):
    for test, output in error_list:
        self.stream.write(self.sep1)
        self.stream.write(f'{label}: {test.id()}\n')
        self.stream.write(self.sep2)
        self.stream.write(output)