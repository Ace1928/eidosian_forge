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
def _parse_pkcs12_35_0_0(pkcs12_bytes, passphrase=None):
    private_key, certificate, additional_certificates = _load_key_and_certificates(pkcs12_bytes, passphrase)
    friendly_name = None
    if certificate:
        backend = default_backend()
        pkcs12 = backend._ffi.gc(backend._lib.d2i_PKCS12_bio(backend._bytes_to_bio(pkcs12_bytes).bio, backend._ffi.NULL), backend._lib.PKCS12_free)
        certificate_x509_ptr = backend._ffi.new('X509 **')
        with backend._zeroed_null_terminated_buf(to_bytes(passphrase) if passphrase is not None else None) as passphrase_buffer:
            backend._lib.PKCS12_parse(pkcs12, passphrase_buffer, backend._ffi.new('EVP_PKEY **'), certificate_x509_ptr, backend._ffi.new('Cryptography_STACK_OF_X509 **'))
        if certificate_x509_ptr[0] != backend._ffi.NULL:
            maybe_name = backend._lib.X509_alias_get0(certificate_x509_ptr[0], backend._ffi.NULL)
            if maybe_name != backend._ffi.NULL:
                friendly_name = backend._ffi.string(maybe_name)
    return (private_key, certificate, additional_certificates, friendly_name)