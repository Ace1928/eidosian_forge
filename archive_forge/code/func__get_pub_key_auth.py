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
def _get_pub_key_auth(pub_key: bytes, usage: str, nonce: typing.Optional[bytes]=None) -> bytes:
    """Computes the public key authentication value.

    Params:
        pub_key: The public key to transform.
        usage: Either `initiate` or `accept` to denote if the key is for the
            client to server or vice versa.
        nonce: A 32 byte nonce used for CredSSP version 5 or newer.

    Returns:
        bytes: The public key authentication value.
    """
    if nonce:
        direction = b'Client-To-Server' if usage == 'initiate' else b'Server-To-Client'
        hash_input = b'CredSSP %s Binding Hash\x00' % direction + nonce + pub_key
        key_auth = hashlib.sha256(hash_input).digest()
    elif usage == 'accept':
        first_byte = struct.unpack('B', pub_key[0:1])[0]
        key_auth = struct.pack('B', first_byte + 1) + pub_key[1:]
    else:
        key_auth = pub_key
    return key_auth