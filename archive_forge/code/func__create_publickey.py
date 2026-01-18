from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.io import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.publickey_info import (
def _create_publickey(self, module):
    self.privatekey = load_privatekey(path=self.privatekey_path, content=self.privatekey_content, passphrase=self.privatekey_passphrase, backend=self.backend)
    if self.backend == 'cryptography':
        if self.format == 'OpenSSH':
            return self.privatekey.public_key().public_bytes(crypto_serialization.Encoding.OpenSSH, crypto_serialization.PublicFormat.OpenSSH)
        else:
            return self.privatekey.public_key().public_bytes(crypto_serialization.Encoding.PEM, crypto_serialization.PublicFormat.SubjectPublicKeyInfo)