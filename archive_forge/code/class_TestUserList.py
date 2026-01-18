from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient import common
from troveclient.osc.v1 import database_users
from troveclient.tests.osc.v1 import fakes
class TestUserList(TestUsers):
    columns = database_users.ListDatabaseUsers.columns
    values = ('harry', '%', 'db1')

    def setUp(self):
        super(TestUserList, self).setUp()
        self.cmd = database_users.ListDatabaseUsers(self.app, None)
        data = [self.fake_users.get_instances_1234_users_harry()]
        self.user_client.list.return_value = common.Paginated(data)

    @mock.patch.object(utils, 'find_resource')
    def test_user_list_defaults(self, mock_find):
        args = ['my_instance']
        mock_find.return_value = args[0]
        parsed_args = self.check_parser(self.cmd, args, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.user_client.list.assert_called_once_with(*args)
        self.assertEqual(self.columns, columns)
        self.assertEqual([self.values], data)