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
def _check_authority_key_identifier(extensions):
    ext = _find_extension(extensions, cryptography.x509.AuthorityKeyIdentifier)
    if self.authority_key_identifier is not None or self.authority_cert_issuer is not None or self.authority_cert_serial_number is not None:
        if not ext or ext.critical:
            return False
        aci = None
        csr_aci = None
        if self.authority_cert_issuer is not None:
            aci = [to_text(cryptography_get_name(n, 'authority cert issuer')) for n in self.authority_cert_issuer]
        if ext.value.authority_cert_issuer is not None:
            csr_aci = [to_text(n) for n in ext.value.authority_cert_issuer]
        return ext.value.key_identifier == self.authority_key_identifier and csr_aci == aci and (ext.value.authority_cert_serial_number == self.authority_cert_serial_number)
    else:
        return ext is None