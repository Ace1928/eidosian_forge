from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient import common
from troveclient.osc.v1 import database_users
from troveclient.tests.osc.v1 import fakes
class TestDatabaseUserDelete(TestUsers):

    def setUp(self):
        super(TestDatabaseUserDelete, self).setUp()
        self.cmd = database_users.DeleteDatabaseUser(self.app, None)

    @mock.patch.object(utils, 'find_resource')
    def test_user_delete(self, mock_find):
        args = ['userinstance', 'user1', '--host', '1.1.1.1']
        mock_find.return_value = args[0]
        parsed_args = self.check_parser(self.cmd, args, [])
        result = self.cmd.take_action(parsed_args)
        self.user_client.delete.assert_called_with('userinstance', 'user1', '1.1.1.1')
        self.assertIsNone(result)

    @mock.patch.object(utils, 'find_resource')
    def test_user_delete_without_host(self, mock_find):
        args = ['userinstance2', 'user1']
        mock_find.return_value = args[0]
        parsed_args = self.check_parser(self.cmd, args, [])
        result = self.cmd.take_action(parsed_args)
        self.user_client.delete.assert_called_with('userinstance2', 'user1', None)
        self.assertIsNone(result)

    @mock.patch.object(utils, 'find_resource')
    def test_user_delete_with_exception(self, mock_find):
        args = ['userfakeinstance', 'db1', '--host', '1.1.1.1']
        parsed_args = self.check_parser(self.cmd, args, [])
        mock_find.side_effect = exceptions.CommandError
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)