from unittest import mock
from castellan.tests.unit.key_manager import fake
from os_brick import encryptors
from os_brick.tests import base
def _test_get_encryptor(self, provider, expected_provider_class):
    encryption = {'control_location': 'front-end', 'provider': provider}
    encryptor = encryptors.get_volume_encryptor(root_helper=self.root_helper, connection_info=self.connection_info, keymgr=self.keymgr, **encryption)
    self.assertIsInstance(encryptor, expected_provider_class)