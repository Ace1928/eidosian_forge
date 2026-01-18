import datetime
from oslo_utils import timeutils
from novaclient.tests.functional import base
class TestServersBootNovaClient(base.ClientTestBase):
    """Servers boot functional tests."""
    COMPUTE_API_VERSION = '2.1'

    def _boot_server_with_legacy_bdm(self, bdm_params=()):
        volume_size = 1
        volume_name = self.name_generate()
        volume = self.cinder.volumes.create(size=volume_size, name=volume_name, imageRef=self.image.id)
        self.wait_for_volume_status(volume, 'available')
        if len(bdm_params) >= 3 and bdm_params[2] == '1':
            delete_volume = False
        else:
            delete_volume = True
        bdm_params = ':'.join(bdm_params)
        if bdm_params:
            bdm_params = ''.join((':', bdm_params))
        params = '%(name)s --flavor %(flavor)s --poll --block-device-mapping vda=%(volume_id)s%(bdm_params)s' % {'name': self.name_generate(), 'flavor': self.flavor.id, 'volume_id': volume.id, 'bdm_params': bdm_params}
        if self.multiple_networks:
            params += ' --nic net-id=%s' % self.network.id
        server_info = self.nova('boot', params=params)
        server_id = self._get_value_from_the_table(server_info, 'id')
        self.client.servers.delete(server_id)
        self.wait_for_resource_delete(server_id, self.client.servers)
        if delete_volume:
            self.cinder.volumes.delete(volume.id)
            self.wait_for_resource_delete(volume.id, self.cinder.volumes)

    def test_boot_server_with_legacy_bdm(self):
        params = ('', '', '1')
        self._boot_server_with_legacy_bdm(bdm_params=params)

    def test_boot_server_with_legacy_bdm_volume_id_only(self):
        self._boot_server_with_legacy_bdm()

    def test_boot_server_with_net_name(self):
        server_info = self.nova('boot', params='%(name)s --flavor %(flavor)s --image %(image)s --poll --nic net-name=%(net-name)s' % {'name': self.name_generate(), 'image': self.image.id, 'flavor': self.flavor.id, 'net-name': self.network.name})
        server_id = self._get_value_from_the_table(server_info, 'id')
        self.client.servers.delete(server_id)
        self.wait_for_resource_delete(server_id, self.client.servers)

    def test_boot_server_using_image_with(self):
        """Scenario test which does the following:

        1. Create a server.
        2. Create a snapshot image of the server with a special meta key.
        3. Create a second server using the --image-with option using the meta
           key stored in the snapshot image created in step 2.
        """
        server_info = self.nova('boot', params='--flavor %(flavor)s --image %(image)s --poll image-with-server-1' % {'image': self.image.id, 'flavor': self.flavor.id})
        server_id = self._get_value_from_the_table(server_info, 'id')
        self.addCleanup(self._cleanup_server, server_id)
        snapshot_info = self.nova('image-create', params='--metadata image_with_meta=%(meta_value)s --show --poll %(server_id)s image-with-snapshot' % {'meta_value': server_id, 'server_id': server_id})
        snapshot_id = self._get_value_from_the_table(snapshot_info, 'id')
        self.addCleanup(self.glance.images.delete, snapshot_id)
        meta_value = self._get_value_from_the_table(snapshot_info, 'image_with_meta')
        self.assertEqual(server_id, meta_value)
        server_info = self.nova('boot', params='--flavor %(flavor)s --image-with image_with_meta=%(meta_value)s --poll image-with-server-2' % {'meta_value': server_id, 'flavor': self.flavor.id})
        server_id = self._get_value_from_the_table(server_info, 'id')
        self.addCleanup(self._cleanup_server, server_id)