import __future__
import difflib
import inspect
import linecache
import os
import pdb
import re
import sys
import traceback
import unittest
from io import StringIO, IncrementalNewlineDecoder
from collections import namedtuple
class SkipDocTestCase(DocTestCase):

    def __init__(self, module):
        self.module = module
        DocTestCase.__init__(self, None)

    def setUp(self):
        self.skipTest('DocTestSuite will not work with -O2 and above')

    def test_skip(self):
        pass

    def shortDescription(self):
        return 'Skipping tests from %s' % self.module.__name__
    __str__ = shortDescription