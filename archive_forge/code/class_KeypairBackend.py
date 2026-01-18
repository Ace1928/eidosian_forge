from __future__ import absolute_import, division, print_function
import abc
import os
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.openssh.cryptography import (
from ansible_collections.community.crypto.plugins.module_utils.openssh.backends.common import (
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
@six.add_metaclass(abc.ABCMeta)
class KeypairBackend(OpensshModule):

    def __init__(self, module):
        super(KeypairBackend, self).__init__(module)
        self.comment = self.module.params['comment']
        self.private_key_path = self.module.params['path']
        self.public_key_path = self.private_key_path + '.pub'
        self.regenerate = self.module.params['regenerate'] if not self.module.params['force'] else 'always'
        self.state = self.module.params['state']
        self.type = self.module.params['type']
        self.size = self._get_size(self.module.params['size'])
        self._validate_path()
        self.original_private_key = None
        self.original_public_key = None
        self.private_key = None
        self.public_key = None

    def _get_size(self, size):
        if self.type in ('rsa', 'rsa1'):
            result = 4096 if size is None else size
            if result < 1024:
                return self.module.fail_json(msg='For RSA keys, the minimum size is 1024 bits and the default is 4096 bits. ' + 'Attempting to use bit lengths under 1024 will cause the module to fail.')
        elif self.type == 'dsa':
            result = 1024 if size is None else size
            if result != 1024:
                return self.module.fail_json(msg='DSA keys must be exactly 1024 bits as specified by FIPS 186-2.')
        elif self.type == 'ecdsa':
            result = 256 if size is None else size
            if result not in (256, 384, 521):
                return self.module.fail_json(msg='For ECDSA keys, size determines the key length by selecting from one of ' + 'three elliptic curve sizes: 256, 384 or 521 bits. ' + 'Attempting to use bit lengths other than these three values for ECDSA keys will ' + 'cause this module to fail.')
        elif self.type == 'ed25519':
            result = 256
        else:
            return self.module.fail_json(msg='%s is not a valid value for key type' % self.type)
        return result

    def _validate_path(self):
        self._check_if_base_dir(self.private_key_path)
        if os.path.isdir(self.private_key_path):
            self.module.fail_json(msg='%s is a directory. Please specify a path to a file.' % self.private_key_path)

    def _execute(self):
        self.original_private_key = self._load_private_key()
        self.original_public_key = self._load_public_key()
        if self.state == 'present':
            self._validate_key_load()
            if self._should_generate():
                self._generate()
            elif not self._public_key_valid():
                self._restore_public_key()
            self.private_key = self._load_private_key()
            self.public_key = self._load_public_key()
            for path in (self.private_key_path, self.public_key_path):
                self._update_permissions(path)
        elif self._should_remove():
            self._remove()

    def _load_private_key(self):
        result = None
        if self._private_key_exists():
            try:
                result = self._get_private_key()
            except Exception:
                pass
        return result

    def _private_key_exists(self):
        return os.path.exists(self.private_key_path)

    @abc.abstractmethod
    def _get_private_key(self):
        pass

    def _load_public_key(self):
        result = None
        if self._public_key_exists():
            try:
                result = PublicKey.load(self.public_key_path)
            except (IOError, OSError):
                pass
        return result

    def _public_key_exists(self):
        return os.path.exists(self.public_key_path)

    def _validate_key_load(self):
        if self._private_key_exists() and self.regenerate in ('never', 'fail', 'partial_idempotence') and (self.original_private_key is None or not self._private_key_readable()):
            self.module.fail_json(msg='Unable to read the key. The key is protected with a passphrase or broken. ' + 'Will not proceed. To force regeneration, call the module with `generate` ' + 'set to `full_idempotence` or `always`, or with `force=true`.')

    @abc.abstractmethod
    def _private_key_readable(self):
        pass

    def _should_generate(self):
        if self.original_private_key is None:
            return True
        elif self.regenerate == 'never':
            return False
        elif self.regenerate == 'fail':
            if not self._private_key_valid():
                self.module.fail_json(msg='Key has wrong type and/or size. Will not proceed. ' + 'To force regeneration, call the module with `generate` set to ' + '`partial_idempotence`, `full_idempotence` or `always`, or with `force=true`.')
            return False
        elif self.regenerate in ('partial_idempotence', 'full_idempotence'):
            return not self._private_key_valid()
        else:
            return True

    def _private_key_valid(self):
        if self.original_private_key is None:
            return False
        return all([self.size == self.original_private_key.size, self.type == self.original_private_key.type, self._private_key_valid_backend()])

    @abc.abstractmethod
    def _private_key_valid_backend(self):
        pass

    @OpensshModule.trigger_change
    @OpensshModule.skip_if_check_mode
    def _generate(self):
        temp_private_key, temp_public_key = self._generate_temp_keypair()
        try:
            self._safe_secure_move([(temp_private_key, self.private_key_path), (temp_public_key, self.public_key_path)])
        except OSError as e:
            self.module.fail_json(msg=to_native(e))

    def _generate_temp_keypair(self):
        temp_private_key = os.path.join(self.module.tmpdir, os.path.basename(self.private_key_path))
        temp_public_key = temp_private_key + '.pub'
        try:
            self._generate_keypair(temp_private_key)
        except (IOError, OSError) as e:
            self.module.fail_json(msg=to_native(e))
        for f in (temp_private_key, temp_public_key):
            self.module.add_cleanup_file(f)
        return (temp_private_key, temp_public_key)

    @abc.abstractmethod
    def _generate_keypair(self, private_key_path):
        pass

    def _public_key_valid(self):
        if self.original_public_key is None:
            return False
        valid_public_key = self._get_public_key()
        valid_public_key.comment = self.comment
        return self.original_public_key == valid_public_key

    @abc.abstractmethod
    def _get_public_key(self):
        pass

    @OpensshModule.trigger_change
    @OpensshModule.skip_if_check_mode
    def _restore_public_key(self):
        try:
            temp_public_key = self._create_temp_public_key(str(self._get_public_key()) + '\n')
            self._safe_secure_move([(temp_public_key, self.public_key_path)])
        except (IOError, OSError):
            self.module.fail_json(msg='The public key is missing or does not match the private key. ' + 'Unable to regenerate the public key.')
        if self.comment:
            self._update_comment()

    def _create_temp_public_key(self, content):
        temp_public_key = os.path.join(self.module.tmpdir, os.path.basename(self.public_key_path))
        default_permissions = 420
        existing_permissions = file_mode(self.public_key_path)
        try:
            secure_write(temp_public_key, existing_permissions or default_permissions, to_bytes(content))
        except (IOError, OSError) as e:
            self.module.fail_json(msg=to_native(e))
        self.module.add_cleanup_file(temp_public_key)
        return temp_public_key

    @abc.abstractmethod
    def _update_comment(self):
        pass

    def _should_remove(self):
        return self._private_key_exists() or self._public_key_exists()

    @OpensshModule.trigger_change
    @OpensshModule.skip_if_check_mode
    def _remove(self):
        try:
            if self._private_key_exists():
                os.remove(self.private_key_path)
            if self._public_key_exists():
                os.remove(self.public_key_path)
        except (IOError, OSError) as e:
            self.module.fail_json(msg=to_native(e))

    @property
    def _result(self):
        private_key = self.private_key or self.original_private_key
        public_key = self.public_key or self.original_public_key
        return {'size': self.size, 'type': self.type, 'filename': self.private_key_path, 'fingerprint': private_key.fingerprint if private_key else '', 'public_key': str(public_key) if public_key else '', 'comment': public_key.comment if public_key else ''}

    @property
    def diff(self):
        before = self.original_private_key.to_dict() if self.original_private_key else {}
        before.update(self.original_public_key.to_dict() if self.original_public_key else {})
        after = self.private_key.to_dict() if self.private_key else {}
        after.update(self.public_key.to_dict() if self.public_key else {})
        return {'before': before, 'after': after}