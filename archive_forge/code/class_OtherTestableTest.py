import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
class OtherTestableTest(unittest.TestCase):

    def setUp(self):
        ordering.append('setUp2')
        self.addCleanup(cleanup3)

    def testNothing(self):
        ordering.append('test2')

    def tearDown(self):
        ordering.append('tearDown2')