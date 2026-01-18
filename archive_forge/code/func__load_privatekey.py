from __future__ import absolute_import, division, print_function
import abc
import base64
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.privatekey_info import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.common import ArgumentSpec
def _load_privatekey(self):
    data = self.existing_private_key_bytes
    try:
        format = identify_private_key_format(data)
        if format == 'raw':
            if len(data) == 56 and CRYPTOGRAPHY_HAS_X448:
                return cryptography.hazmat.primitives.asymmetric.x448.X448PrivateKey.from_private_bytes(data)
            if len(data) == 57 and CRYPTOGRAPHY_HAS_ED448:
                return cryptography.hazmat.primitives.asymmetric.ed448.Ed448PrivateKey.from_private_bytes(data)
            if len(data) == 32:
                if CRYPTOGRAPHY_HAS_X25519 and (self.type == 'X25519' or not CRYPTOGRAPHY_HAS_ED25519):
                    return cryptography.hazmat.primitives.asymmetric.x25519.X25519PrivateKey.from_private_bytes(data)
                if CRYPTOGRAPHY_HAS_ED25519 and (self.type == 'Ed25519' or not CRYPTOGRAPHY_HAS_X25519):
                    return cryptography.hazmat.primitives.asymmetric.ed25519.Ed25519PrivateKey.from_private_bytes(data)
                if CRYPTOGRAPHY_HAS_X25519 and CRYPTOGRAPHY_HAS_ED25519:
                    try:
                        return cryptography.hazmat.primitives.asymmetric.x25519.X25519PrivateKey.from_private_bytes(data)
                    except Exception:
                        return cryptography.hazmat.primitives.asymmetric.ed25519.Ed25519PrivateKey.from_private_bytes(data)
            raise PrivateKeyError('Cannot load raw key')
        else:
            return cryptography.hazmat.primitives.serialization.load_pem_private_key(data, None if self.passphrase is None else to_bytes(self.passphrase), backend=self.cryptography_backend)
    except Exception as e:
        raise PrivateKeyError(e)