import base64
import os
import stat
from cryptography import fernet
from oslo_log import log
from keystone.common import utils
import keystone.conf
def _create_tmp_new_key(self, keystone_user_id, keystone_group_id):
    """Securely create a new tmp encryption key.

        This created key is not effective until _become_valid_new_key().
        """
    key = fernet.Fernet.generate_key()
    old_umask = os.umask(127)
    if keystone_user_id and keystone_group_id:
        old_egid = os.getegid()
        old_euid = os.geteuid()
        os.setegid(keystone_group_id)
        os.seteuid(keystone_user_id)
    elif keystone_user_id or keystone_group_id:
        LOG.warning('Unable to change the ownership of the new key without a keystone user ID and keystone group ID both being provided: %s', self.key_repository)
    key_file = os.path.join(self.key_repository, '0.tmp')
    create_success = False
    try:
        with open(key_file, 'w') as f:
            f.write(key.decode('utf-8'))
            f.flush()
            create_success = True
    except IOError:
        LOG.error('Failed to create new temporary key: %s', key_file)
        raise
    finally:
        os.umask(old_umask)
        if keystone_user_id and keystone_group_id:
            os.seteuid(old_euid)
            os.setegid(old_egid)
        if not create_success and os.access(key_file, os.F_OK):
            os.remove(key_file)
    LOG.info('Created a new temporary key: %s', key_file)