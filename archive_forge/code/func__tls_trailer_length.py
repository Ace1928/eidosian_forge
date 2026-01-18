import base64
import hashlib
import logging
import os
import re
import shutil
import ssl
import struct
import tempfile
import typing
import spnego
from spnego._context import (
from spnego._credential import Credential, Password, unify_credentials
from spnego._credssp_structures import (
from spnego._text import to_text
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import (
from spnego.tls import (
def _tls_trailer_length(data_length: int, protocol: str, cipher_suite: str) -> int:
    """Gets the length of the TLS trailer.

    WinRM wrapping needs to split the trailer/header with the data but the
    length of the trailer is dependent on the cipher suite that was negotiated.
    On Windows you can get this length by calling `QueryContextAttributes`_
    with the `SecPkgContext_StreamSizes`_ structure. Unfortunately we need to
    work on other platforms so we calculate it manually.

    Params:
        data_length: The length of the TLS data used to calculate the padding
            size.
        protocol: The TLS protocol negotiated between the client and server.
        cipher_suite: The TLS cipher suite negotiated between the client and
            server.

    Returns:
        int: The length of the trailer.

    .. _QueryContextAttributes:
        https://docs.microsoft.com/en-us/windows/win32/api/sspi/nf-sspi-querycontextattributesw

    .. _SecPkgContext_StreamSizes:
        https://docs.microsoft.com/en-us/windows/win32/api/sspi/ns-sspi-secpkgcontext_streamsizes
    """
    if protocol == 'TLSv1.3':
        trailer_length = 17
    elif re.match('^.*[-_]GCM[-_][\\w\\d]*$', cipher_suite):
        trailer_length = 16
    else:
        hash_algorithm = cipher_suite.split('-')[-1]
        hash_length = {'MD5': 16, 'SHA': 20, 'SHA256': 32, 'SHA384': 48}.get(hash_algorithm, 0)
        pre_pad_length = data_length + hash_length
        if 'RC4' in cipher_suite:
            padding_length = 0
        elif 'DES' in cipher_suite or '3DES' in cipher_suite:
            padding_length = 8 - pre_pad_length % 8
        else:
            padding_length = 16 - pre_pad_length % 16
        trailer_length = pre_pad_length + padding_length - data_length
    return trailer_length