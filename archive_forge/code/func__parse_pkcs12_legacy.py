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
def _parse_pkcs12_legacy(pkcs12_bytes, passphrase=None):
    private_key, certificate, additional_certificates = _load_key_and_certificates(pkcs12_bytes, passphrase)
    friendly_name = None
    if certificate:
        backend = certificate._backend
        maybe_name = backend._lib.X509_alias_get0(certificate._x509, backend._ffi.NULL)
        if maybe_name != backend._ffi.NULL:
            friendly_name = backend._ffi.string(maybe_name)
    return (private_key, certificate, additional_certificates, friendly_name)