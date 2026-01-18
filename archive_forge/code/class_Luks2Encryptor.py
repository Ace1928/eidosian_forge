import binascii
import os
from oslo_concurrency import processutils
from oslo_log import log as logging
from os_brick.encryptors import base
from os_brick import exception
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
class Luks2Encryptor(LuksEncryptor):
    """A VolumeEncryptor based on LUKS v2.

    This VolumeEncryptor uses dm-crypt to encrypt the specified volume.
    """

    def __init__(self, root_helper, connection_info, keymgr, execute=None, *args, **kwargs):
        super(Luks2Encryptor, self).__init__(*args, root_helper=root_helper, connection_info=connection_info, keymgr=keymgr, execute=execute, **kwargs)

    def _format_volume(self, passphrase, **kwargs):
        """Creates a LUKS v2 header on the volume.

        :param passphrase: the passphrase used to access the volume
        """
        self._format_luks_volume(passphrase, 'luks2', **kwargs)