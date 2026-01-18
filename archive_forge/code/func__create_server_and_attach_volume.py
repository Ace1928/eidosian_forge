from oslo_serialization import jsonutils
from novaclient.tests.functional import base
def _create_server_and_attach_volume(self):
    server = self._create_server()
    volume = self.cinder.volumes.create(1)
    self.addCleanup(volume.delete)
    self.wait_for_volume_status(volume, 'available')
    self.nova('volume-attach', params='%s %s' % (server.name, volume.id))
    self.addCleanup(self._release_volume, server, volume)
    self.wait_for_volume_status(volume, 'in-use')
    return (server, volume)