from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import datastores
from troveclient.tests.osc.v1 import fakes
class TestDatastoreVersionList(TestDatastores):
    columns = datastores.ListDatastoreVersions.columns
    values = ('v-56', '5.6', '')

    def setUp(self):
        super(TestDatastoreVersionList, self).setUp()
        self.cmd = datastores.ListDatastoreVersions(self.app, None)
        self.data = [self.fake_datastores.get_datastores_d_123_versions()]
        self.datastore_version_client.list.return_value = common.Paginated(self.data)

    def test_datastore_version_list_defaults(self):
        args = ['mysql']
        parsed_args = self.check_parser(self.cmd, args, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.datastore_version_client.list.assert_called_once_with(args[0])
        self.assertEqual(self.columns, columns)
        self.assertEqual([self.values], data)