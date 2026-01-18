import os
import testresources
import testscenarios
import unittest
from unittest import mock
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.tests import base as test_base
class TestWScenarios(unittest.TestCase):
    """a 'do nothing' test suite.

    Should generate exactly four tests when testscenarios is used.

    """

    def test_one(self):
        pass

    def test_two(self):
        pass
    scenarios = [('scenario1', dict(scenario='scenario 1')), ('scenario2', dict(scenario='scenario 2'))]