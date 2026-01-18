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
def _check_passphrase(self):
    try:
        format = identify_private_key_format(self.existing_private_key_bytes)
        if format == 'raw':
            self._load_privatekey()
            return self.passphrase is None
        else:
            return cryptography.hazmat.primitives.serialization.load_pem_private_key(self.existing_private_key_bytes, None if self.passphrase is None else to_bytes(self.passphrase), backend=self.cryptography_backend)
    except Exception as dummy:
        return False