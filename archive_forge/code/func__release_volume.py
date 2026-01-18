from oslo_serialization import jsonutils
from novaclient.tests.functional import base
def _release_volume(self, server, volume):
    self.nova('volume-detach', params='%s %s' % (server.id, volume.id))
    self.wait_for_volume_status(volume, 'available')