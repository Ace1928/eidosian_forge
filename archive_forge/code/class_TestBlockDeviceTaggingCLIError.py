from tempest.lib import exceptions
from novaclient.tests.functional import base
class TestBlockDeviceTaggingCLIError(base.ClientTestBase):
    """Negative test that asserts that creating a server with a tagged
    block device with a specific microversion will fail.
    """
    COMPUTE_API_VERSION = '2.31'

    def test_boot_server_with_tagged_block_devices_with_error(self):
        try:
            output = self.nova('boot', params='%(name)s --flavor %(flavor)s --poll --nic net-id=%(net-uuid)s --block-device source=image,dest=volume,id=%(image)s,size=1,bootindex=0,shutdown=remove,tag=bar' % {'name': self.name_generate(), 'flavor': self.flavor.id, 'net-uuid': self.network.id, 'image': self.image.id})
        except exceptions.CommandFailed as e:
            self.assertIn("ERROR (CommandError): 'tag' in block device mapping is not supported in API version %s." % self.COMPUTE_API_VERSION, str(e))
        else:
            server_id = self._get_value_from_the_table(output, 'id')
            self.client.servers.delete(server_id)
            self.wait_for_resource_delete(server_id, self.client.servers)
            self.fail('Booting a server with block device tag is not failed.')