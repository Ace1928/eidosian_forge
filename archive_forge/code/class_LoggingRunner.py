import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
class LoggingRunner(unittest.TextTestRunner):

    def __init__(self, events):
        super(LoggingRunner, self).__init__(io.StringIO())
        self._events = events

    def _makeResult(self):
        return LoggingTextResult(self._events)