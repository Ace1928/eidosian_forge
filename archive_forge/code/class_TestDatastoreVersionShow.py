from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import datastores
from troveclient.tests.osc.v1 import fakes
class TestDatastoreVersionShow(TestDatastores):
    values = ('v-56', '5.6')

    def setUp(self):
        super(TestDatastoreVersionShow, self).setUp()
        self.cmd = datastores.ShowDatastoreVersion(self.app, None)
        self.data = self.fake_datastores.get_datastores_d_123_versions()
        self.datastore_version_client.get.return_value = self.data
        self.columns = ('id', 'name')

    def test_datastore_version_show_defaults(self):
        args = ['5.6', '--datastore', 'mysql']
        verifylist = [('datastore_version', '5.6'), ('datastore', 'mysql')]
        parsed_args = self.check_parser(self.cmd, args, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.values, data)

    def test_datastore_version_show_with_version_id_exception(self):
        args = ['v-56']
        verifylist = [('datastore_version', 'v-56')]
        parsed_args = self.check_parser(self.cmd, args, verifylist)
        self.assertRaises(exceptions.NoUniqueMatch, self.cmd.take_action, parsed_args)