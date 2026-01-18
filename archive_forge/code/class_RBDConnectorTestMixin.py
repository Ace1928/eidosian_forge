from unittest import mock
import ddt
from os_brick.initiator.connectors import base_rbd
from os_brick.tests import base
class RBDConnectorTestMixin(object):

    def setUp(self):
        super(RBDConnectorTestMixin, self).setUp()
        self.user = 'fake_user'
        self.pool = 'fake_pool'
        self.volume = 'fake_volume'
        self.clustername = 'fake_ceph'
        self.hosts = ['192.168.10.2']
        self.ports = ['6789']
        self.keyring = '[client.cinder]\n  key = test\n'
        self.image_name = '%s/%s' % (self.pool, self.volume)
        self.connection_properties = {'auth_username': self.user, 'name': self.image_name, 'cluster_name': self.clustername, 'hosts': self.hosts, 'ports': self.ports, 'keyring': self.keyring}