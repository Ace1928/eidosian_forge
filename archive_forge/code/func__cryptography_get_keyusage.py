from __future__ import absolute_import, division, print_function
import base64
import binascii
import re
import sys
import traceback
from ansible.module_utils.common.text.converters import to_text, to_bytes, to_native
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlunparse, ParseResult
from ._asn1 import serialize_asn1_string_as_der
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import missing_required_lib
from .basic import (
from ._objects import (
from ._obj2txt import obj2txt
def _cryptography_get_keyusage(usage):
    """
    Given a key usage identifier string, returns the parameter name used by cryptography's x509.KeyUsage().
    Raises an OpenSSLObjectError if the identifier is unknown.
    """
    if usage in ('Digital Signature', 'digitalSignature'):
        return 'digital_signature'
    if usage in ('Non Repudiation', 'nonRepudiation'):
        return 'content_commitment'
    if usage in ('Key Encipherment', 'keyEncipherment'):
        return 'key_encipherment'
    if usage in ('Data Encipherment', 'dataEncipherment'):
        return 'data_encipherment'
    if usage in ('Key Agreement', 'keyAgreement'):
        return 'key_agreement'
    if usage in ('Certificate Sign', 'keyCertSign'):
        return 'key_cert_sign'
    if usage in ('CRL Sign', 'cRLSign'):
        return 'crl_sign'
    if usage in ('Encipher Only', 'encipherOnly'):
        return 'encipher_only'
    if usage in ('Decipher Only', 'decipherOnly'):
        return 'decipher_only'
    raise OpenSSLObjectError('Unknown key usage "{0}"'.format(usage))