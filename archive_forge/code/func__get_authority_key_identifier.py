from __future__ import absolute_import, division, print_function
import abc
import binascii
import datetime
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.publickey_info import (
def _get_authority_key_identifier(self):
    try:
        ext = self.cert.extensions.get_extension_for_class(x509.AuthorityKeyIdentifier)
        issuer = None
        if ext.value.authority_cert_issuer is not None:
            issuer = [cryptography_decode_name(san, idn_rewrite=self.name_encoding) for san in ext.value.authority_cert_issuer]
        return (ext.value.key_identifier, issuer, ext.value.authority_cert_serial_number)
    except cryptography.x509.ExtensionNotFound:
        return (None, None, None)