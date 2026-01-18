import doctest
from pprint import pformat
import unittest
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import DocTestMatches, Equals
from testtools.testresult.doubles import StreamResult as LoggingStream
from testtools.testsuite import FixtureSuite, sorted_tests
from testtools.tests.helpers import LoggingResult
class Skips(TestCase):

    @classmethod
    def setUpClass(cls):
        raise cls.skipException('foo')

    def test_notrun(self):
        pass