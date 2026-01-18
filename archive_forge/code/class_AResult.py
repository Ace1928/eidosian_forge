import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
class AResult(unittest.TestResult):

    def __init__(self, stream, descriptions, verbosity):
        super(AResult, self).__init__(stream, descriptions, verbosity)