import base64
import os
import stat
from cryptography import fernet
from oslo_log import log
from keystone.common import utils
import keystone.conf
def _create_new_key(self, keystone_user_id, keystone_group_id):
    """Securely create a new encryption key.

        Create a new key that is readable by the Keystone group and Keystone
        user.

        To avoid disk write failure, this function will create a tmp key file
        first, and then rename it as the valid new key.
        """
    self._create_tmp_new_key(keystone_user_id, keystone_group_id)
    self._become_valid_new_key()