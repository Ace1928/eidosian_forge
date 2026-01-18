from os_brick.encryptors import nop
from os_brick.tests.encryptors import test_base
class NoOpEncryptorTestCase(test_base.VolumeEncryptorTestCase):

    def _create(self):
        return nop.NoOpEncryptor(root_helper=self.root_helper, connection_info=self.connection_info, keymgr=self.keymgr)

    def test_attach_volume(self):
        test_args = {'control_location': 'front-end', 'provider': 'NoOpEncryptor'}
        self.encryptor.attach_volume(None, **test_args)

    def test_detach_volume(self):
        test_args = {'control_location': 'front-end', 'provider': 'NoOpEncryptor'}
        self.encryptor.detach_volume(**test_args)

    def test_extend_volume(self):
        self.encryptor.extend_volume('context', anything=1, goes='asdf')