import os.path
from os.path import abspath
import re
import sys
import types
import pickle
from test import support
from test.support import import_helper
import unittest
import unittest.mock
import unittest.test
class TestableTestProgram(unittest.TestProgram):
    module = None
    exit = True
    defaultTest = failfast = catchbreak = buffer = None
    verbosity = 1
    progName = ''
    testRunner = testLoader = None

    def __init__(self):
        pass