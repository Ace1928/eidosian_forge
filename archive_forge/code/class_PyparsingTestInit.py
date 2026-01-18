from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class PyparsingTestInit(ParseTestCase):

    def setUp(self):
        from pyparsing import __version__ as pyparsingVersion, __versionTime__ as pyparsingVersionTime
        print_('Beginning test of pyparsing, version', pyparsingVersion, pyparsingVersionTime)
        print_('Python version', sys.version)

    def tearDown(self):
        pass