import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
class OuterTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ordering.append('outer setup')
        cls.addClassCleanup(ordering.append, 'outer cleanup')

    def test(self):
        ordering.append('start outer test')
        runTests(InnerTest)
        ordering.append('end outer test')