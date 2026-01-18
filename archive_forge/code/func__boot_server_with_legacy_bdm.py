import datetime
from oslo_utils import timeutils
from novaclient.tests.functional import base
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