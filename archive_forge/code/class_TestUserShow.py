from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient import common
from troveclient.osc.v1 import database_users
from troveclient.tests.osc.v1 import fakes
class TestUserShow(TestUsers):
    values = ([{'name': 'db1'}], '%', 'harry')

    def setUp(self):
        super(TestUserShow, self).setUp()
        self.cmd = database_users.ShowDatabaseUser(self.app, None)
        self.data = self.fake_users.get_instances_1234_users_harry()
        self.user_client.get.return_value = self.data
        self.columns = ('databases', 'host', 'name')

    @mock.patch.object(utils, 'find_resource')
    def test_user_show_defaults(self, mock_find):
        args = ['my_instance', 'harry']
        mock_find.return_value = args[0]
        parsed_args = self.check_parser(self.cmd, args, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.values, data)