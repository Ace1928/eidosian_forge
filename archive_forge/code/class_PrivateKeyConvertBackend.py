from __future__ import absolute_import, division, print_function
import abc
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.io import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.common import ArgumentSpec
@six.add_metaclass(abc.ABCMeta)
class PrivateKeyConvertBackend:

    def __init__(self, module, backend):
        self.module = module
        self.src_path = module.params['src_path']
        self.src_content = module.params['src_content']
        self.src_passphrase = module.params['src_passphrase']
        self.format = module.params['format']
        self.dest_passphrase = module.params['dest_passphrase']
        self.backend = backend
        self.src_private_key = None
        if self.src_path is not None:
            self.src_private_key_bytes = load_file(self.src_path, module)
        else:
            self.src_private_key_bytes = self.src_content.encode('utf-8')
        self.dest_private_key = None
        self.dest_private_key_bytes = None

    @abc.abstractmethod
    def get_private_key_data(self):
        """Return bytes for self.src_private_key in output format."""
        pass

    def set_existing_destination(self, privatekey_bytes):
        """Set existing private key bytes. None indicates that the key does not exist."""
        self.dest_private_key_bytes = privatekey_bytes

    def has_existing_destination(self):
        """Query whether an existing private key is/has been there."""
        return self.dest_private_key_bytes is not None

    @abc.abstractmethod
    def _load_private_key(self, data, passphrase, current_hint=None):
        """Check whether data can be loaded as a private key with the provided passphrase. Return tuple (type, private_key)."""
        pass

    def needs_conversion(self):
        """Check whether a conversion is necessary. Must only be called if needs_regeneration() returned False."""
        dummy, self.src_private_key = self._load_private_key(self.src_private_key_bytes, self.src_passphrase)
        if not self.has_existing_destination():
            return True
        try:
            format, self.dest_private_key = self._load_private_key(self.dest_private_key_bytes, self.dest_passphrase, current_hint=self.src_private_key)
        except Exception:
            return True
        return format != self.format or not cryptography_compare_private_keys(self.dest_private_key, self.src_private_key)

    def dump(self):
        """Serialize the object into a dictionary."""
        return {}