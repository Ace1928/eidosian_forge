from __future__ import absolute_import, division, print_function
import abc
import binascii
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_crl import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.csr_info import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.common import ArgumentSpec
def _ensure_private_key_loaded(self):
    """Load the provided private key into self.privatekey."""
    if self.privatekey is not None:
        return
    try:
        self.privatekey = load_privatekey(path=self.privatekey_path, content=self.privatekey_content, passphrase=self.privatekey_passphrase, backend=self.backend)
    except OpenSSLBadPassphraseError as exc:
        raise CertificateSigningRequestError(exc)