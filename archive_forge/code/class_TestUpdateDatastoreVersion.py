from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import datastores
from troveclient.tests.osc.v1 import fakes
class TestUpdateDatastoreVersion(TestDatastores):

    def setUp(self):
        super(TestUpdateDatastoreVersion, self).setUp()
        self.cmd = datastores.UpdateDatastoreVersion(self.app, None)

    def test_update_datastore_version(self):
        version_id = uuidutils.generate_uuid()
        args = [version_id, '--image-tags', 'trove,mysql', '--enable', '--non-default']
        parsed_args = self.check_parser(self.cmd, args, [])
        self.cmd.take_action(parsed_args)
        self.dsversion_mgmt_client.edit.assert_called_once_with(version_id, datastore_manager=None, image=None, active='true', default='false', image_tags=['trove', 'mysql'], name=None)