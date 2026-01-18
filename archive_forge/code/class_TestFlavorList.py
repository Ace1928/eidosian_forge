from troveclient.osc.v1 import database_flavors
from troveclient.tests.osc.v1 import fakes
class TestFlavorList(TestFlavors):
    columns = database_flavors.ListDatabaseFlavors.columns
    values = (1, 'm1.tiny', 512, '', '', '')

    def setUp(self):
        super(TestFlavorList, self).setUp()
        self.cmd = database_flavors.ListDatabaseFlavors(self.app, None)
        self.data = [self.fake_flavors.get_flavors_1()]
        self.flavor_client.list.return_value = self.data
        self.flavor_client.list_datastore_version_associated_flavors.return_value = self.data

    def test_flavor_list_defaults(self):
        parsed_args = self.check_parser(self.cmd, [], [])
        columns, values = self.cmd.take_action(parsed_args)
        self.flavor_client.list.assert_called_once_with()
        self.assertEqual(self.columns, columns)
        self.assertEqual([self.values], values)

    def test_flavor_list_with_optional_args(self):
        args = ['--datastore-type', 'mysql', '--datastore-version-id', '5.6']
        parsed_args = self.check_parser(self.cmd, args, [])
        list_flavor_dict = {'datastore': 'mysql', 'version_id': '5.6'}
        columns, values = self.cmd.take_action(parsed_args)
        self.flavor_client.list_datastore_version_associated_flavors.assert_called_once_with(**list_flavor_dict)
        self.assertEqual(self.columns, columns)
        self.assertEqual([self.values], values)