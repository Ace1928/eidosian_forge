import base64
import os
import stat
from cryptography import fernet
from oslo_log import log
from keystone.common import utils
import keystone.conf
def _become_valid_new_key(self):
    """Make the tmp new key a valid new key.

        The tmp new key must be created by _create_tmp_new_key().
        """
    tmp_key_file = os.path.join(self.key_repository, '0.tmp')
    valid_key_file = os.path.join(self.key_repository, '0')
    os.rename(tmp_key_file, valid_key_file)
    LOG.info('Become a valid new key: %s', valid_key_file)