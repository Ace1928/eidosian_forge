from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import datastores
from troveclient.tests.osc.v1 import fakes
class TestDatastoreShow(TestDatastores):
    values = ('5.6', 'd-123', 'mysql', '5.6 (v-56)')

    def setUp(self):
        super(TestDatastoreShow, self).setUp()
        self.cmd = datastores.ShowDatastore(self.app, None)
        self.data = self.fake_datastores.get_datastores_d_123()
        self.datastore_client.get.return_value = self.data
        self.columns = ('default_version', 'id', 'name', 'versions (id)')

    def test_show(self):
        args = ['mysql']
        parsed_args = self.check_parser(self.cmd, args, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.values, data)