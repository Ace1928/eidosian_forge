import io
import logging
import time
import testtools
from testtools import TestCase
from fixtures import (
class FooFormatter(logging.Formatter):

    def format(self, record):
        self._style = logging.PercentStyle('Foo ' + self._style._fmt)
        self._fmt = self._style._fmt
        return logging.Formatter.format(self, record)