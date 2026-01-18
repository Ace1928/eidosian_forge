import base64
import os
import stat
from cryptography import fernet
from oslo_log import log
from keystone.common import utils
import keystone.conf
class FernetUtils(object):

    def __init__(self, key_repository, max_active_keys, config_group):
        self.key_repository = key_repository
        self.max_active_keys = max_active_keys
        self.config_group = config_group

    def validate_key_repository(self, requires_write=False):
        """Validate permissions on the key repository directory."""
        is_valid = os.access(self.key_repository, os.R_OK) and os.access(self.key_repository, os.X_OK)
        if requires_write:
            is_valid = is_valid and os.access(self.key_repository, os.W_OK)
        if not is_valid:
            LOG.error('Either [%(config_group)s] key_repository does not exist or Keystone does not have sufficient permission to access it: %(key_repo)s', {'key_repo': self.key_repository, 'config_group': self.config_group})
        else:
            stat_info = os.stat(self.key_repository)
            if stat_info.st_mode & stat.S_IROTH or stat_info.st_mode & stat.S_IXOTH:
                LOG.warning('key_repository is world readable: %s', self.key_repository)
        return is_valid

    def create_key_directory(self, keystone_user_id=None, keystone_group_id=None):
        """Attempt to create the key directory if it doesn't exist."""
        utils.create_directory(self.key_repository, keystone_user_id=keystone_user_id, keystone_group_id=keystone_group_id)

    def _create_new_key(self, keystone_user_id, keystone_group_id):
        """Securely create a new encryption key.

        Create a new key that is readable by the Keystone group and Keystone
        user.

        To avoid disk write failure, this function will create a tmp key file
        first, and then rename it as the valid new key.
        """
        self._create_tmp_new_key(keystone_user_id, keystone_group_id)
        self._become_valid_new_key()

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

    def _become_valid_new_key(self):
        """Make the tmp new key a valid new key.

        The tmp new key must be created by _create_tmp_new_key().
        """
        tmp_key_file = os.path.join(self.key_repository, '0.tmp')
        valid_key_file = os.path.join(self.key_repository, '0')
        os.rename(tmp_key_file, valid_key_file)
        LOG.info('Become a valid new key: %s', valid_key_file)

    def _get_key_files(self, key_repo):
        key_files = dict()
        keys = dict()
        for filename in os.listdir(key_repo):
            path = os.path.join(key_repo, str(filename))
            if os.path.isfile(path):
                with open(path, 'r') as key_file:
                    try:
                        key_id = int(filename)
                    except ValueError:
                        pass
                    else:
                        key = key_file.read()
                        if len(key) == 0:
                            LOG.warning('Ignoring empty key found in key repository: %s', path)
                            continue
                        key_files[key_id] = path
                        keys[key_id] = key
        return (key_files, keys)

    def initialize_key_repository(self, keystone_user_id=None, keystone_group_id=None):
        """Create a key repository and bootstrap it with a key.

        :param keystone_user_id: User ID of the Keystone user.
        :param keystone_group_id: Group ID of the Keystone user.

        """
        if os.access(os.path.join(self.key_repository, '0'), os.F_OK):
            LOG.info('Key repository is already initialized; aborting.')
            return
        self._create_new_key(keystone_user_id, keystone_group_id)
        self.rotate_keys(keystone_user_id, keystone_group_id)

    def rotate_keys(self, keystone_user_id=None, keystone_group_id=None):
        """Create a new primary key and revoke excess active keys.

        :param keystone_user_id: User ID of the Keystone user.
        :param keystone_group_id: Group ID of the Keystone user.

        Key rotation utilizes the following behaviors:

        - The highest key number is used as the primary key (used for
          encryption).
        - All keys can be used for decryption.
        - New keys are always created as key "0," which serves as a placeholder
          before promoting it to be the primary key.

        This strategy allows you to safely perform rotation on one node in a
        cluster, before syncing the results of the rotation to all other nodes
        (during both key rotation and synchronization, all nodes must recognize
        all primary keys).

        """
        key_files, _ = self._get_key_files(self.key_repository)
        LOG.info('Starting key rotation with %(count)s key files: %(list)s', {'count': len(key_files), 'list': list(key_files.values())})
        self._create_tmp_new_key(keystone_user_id, keystone_group_id)
        current_primary_key = max(key_files.keys())
        LOG.info('Current primary key is: %s', current_primary_key)
        new_primary_key = current_primary_key + 1
        LOG.info('Next primary key will be: %s', new_primary_key)
        os.rename(os.path.join(self.key_repository, '0'), os.path.join(self.key_repository, str(new_primary_key)))
        key_files.pop(0)
        key_files[new_primary_key] = os.path.join(self.key_repository, str(new_primary_key))
        LOG.info('Promoted key 0 to be the primary: %s', new_primary_key)
        self._become_valid_new_key()
        max_active_keys = self.max_active_keys
        keys = sorted(key_files.keys(), reverse=True)
        while len(keys) > max_active_keys - 1:
            index_to_purge = keys.pop()
            key_to_purge = key_files[index_to_purge]
            LOG.info('Excess key to purge: %s', key_to_purge)
            os.remove(key_to_purge)

    def load_keys(self, use_null_key=False):
        """Load keys from disk into a list.

        The first key in the list is the primary key used for encryption. All
        other keys are active secondary keys that can be used for decrypting
        tokens.

        :param use_null_key: If true, a known key containing null bytes will be
                             appended to the list of returned keys.

        """
        if not self.validate_key_repository():
            if use_null_key:
                return [NULL_KEY]
            return []
        _, keys = self._get_key_files(self.key_repository)
        if len(keys) != self.max_active_keys:
            if self.key_repository == CONF.fernet_tokens.key_repository:
                msg = 'Loaded %(count)d Fernet keys from %(dir)s, but `[fernet_tokens] max_active_keys = %(max)d`; perhaps there have not been enough key rotations to reach `max_active_keys` yet?'
                LOG.debug(msg, {'count': len(keys), 'max': self.max_active_keys, 'dir': self.key_repository})
        key_list = [keys[x] for x in sorted(keys.keys(), reverse=True)]
        if use_null_key:
            key_list.append(NULL_KEY)
        return key_list