import os
import testresources
import testscenarios
import unittest
from unittest import mock
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.tests import base as test_base
def _test_package_level(self, fn):
    load_tests = fn(os.path.join(start_dir, '__init__.py'))
    loader = unittest.TestLoader()
    new_loader = load_tests(loader, unittest.suite.TestSuite(), 'test_fixtures.py')
    self.assertIsInstance(new_loader, testresources.OptimisingTestSuite)
    actual_tests = unittest.TestSuite(testscenarios.generate_scenarios(loader.discover(start_dir, pattern='test_fixtures.py')))
    self.assertEqual(new_loader.countTestCases(), actual_tests.countTestCases())