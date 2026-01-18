import unittest
from testtools.testcase import clone_test_with_new_id
from testscenarios.scenarios import generate_scenarios
class TestWithScenarios(WithScenarios, unittest.TestCase):
    __doc__ = 'Unittest TestCase with support for declarative scenarios.\n    ' + _doc