from unittest import mock
from castellan.tests.unit.key_manager import fake
from os_brick import encryptors
from os_brick.tests import base
class VolumeEncryptorTestCase(base.TestCase):

    def _create(self):
        pass

    def setUp(self):
        super(VolumeEncryptorTestCase, self).setUp()
        self.connection_info = {'data': {'device_path': '/dev/disk/by-path/ip-192.0.2.0:3260-iscsi-iqn.2010-10.org.openstack:volume-fake_uuid-lun-1'}}
        self.root_helper = None
        self.keymgr = fake.fake_api()
        self.encryptor = self._create()