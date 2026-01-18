import testscenarios
from unittest import mock
from cliff import command
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils
class TestGetByPartialName(base.TestBase):

    def setUp(self):
        super(TestGetByPartialName, self).setUp()
        self.commands = {'resource provider list': 1, 'resource class list': 2, 'server list': 3, 'service list': 4}

    def test_no_candidates(self):
        self.assertEqual([], commandmanager._get_commands_by_partial_name(['r', 'p'], self.commands))
        self.assertEqual([], commandmanager._get_commands_by_partial_name(['r', 'p', 'c'], self.commands))

    def test_multiple_candidates(self):
        self.assertEqual(2, len(commandmanager._get_commands_by_partial_name(['se', 'li'], self.commands)))

    def test_one_candidate(self):
        self.assertEqual(['resource provider list'], commandmanager._get_commands_by_partial_name(['r', 'p', 'l'], self.commands))
        self.assertEqual(['resource provider list'], commandmanager._get_commands_by_partial_name(['resource', 'provider', 'list'], self.commands))
        self.assertEqual(['server list'], commandmanager._get_commands_by_partial_name(['serve', 'l'], self.commands))