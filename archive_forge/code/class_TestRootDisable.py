from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient.osc.v1 import database_root
from troveclient.tests.osc.v1 import fakes
class TestRootDisable(TestRoot):

    def setUp(self):
        super(TestRootDisable, self).setUp()
        self.cmd = database_root.DisableDatabaseRoot(self.app, None)
        self.data = self.fake_root.delete_instance_1234_root()

    @mock.patch.object(utils, 'find_resource')
    def test_disable_instance_1234_root(self, mock_find):
        self.root_client.disable_instance_root.return_value = self.data
        args = ['1234']
        mock_find.return_value = args[0]
        parsed_args = self.check_parser(self.cmd, args, [])
        result = self.cmd.take_action(parsed_args)
        self.root_client.disable_instance_root.assert_called_with('1234')
        self.assertIsNone(result)