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
def _get_key_usage(self):
    try:
        current_key_ext = self.cert.extensions.get_extension_for_class(x509.KeyUsage)
        current_key_usage = current_key_ext.value
        key_usage = dict(digital_signature=current_key_usage.digital_signature, content_commitment=current_key_usage.content_commitment, key_encipherment=current_key_usage.key_encipherment, data_encipherment=current_key_usage.data_encipherment, key_agreement=current_key_usage.key_agreement, key_cert_sign=current_key_usage.key_cert_sign, crl_sign=current_key_usage.crl_sign, encipher_only=False, decipher_only=False)
        if key_usage['key_agreement']:
            key_usage.update(dict(encipher_only=current_key_usage.encipher_only, decipher_only=current_key_usage.decipher_only))
        key_usage_names = dict(digital_signature='Digital Signature', content_commitment='Non Repudiation', key_encipherment='Key Encipherment', data_encipherment='Data Encipherment', key_agreement='Key Agreement', key_cert_sign='Certificate Sign', crl_sign='CRL Sign', encipher_only='Encipher Only', decipher_only='Decipher Only')
        return (sorted([key_usage_names[name] for name, value in key_usage.items() if value]), current_key_ext.critical)
    except cryptography.x509.ExtensionNotFound:
        return (None, False)