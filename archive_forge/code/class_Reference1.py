import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
from testscenarios.scenarios import (
class Reference1(unittest.TestCase):
    scenarios = [('1', {}), ('2', {})]

    def test_something(self):
        pass