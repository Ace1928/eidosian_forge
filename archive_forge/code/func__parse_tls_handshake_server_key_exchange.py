import argparse
import base64
import json
import os.path
import re
import struct
import sys
import typing
from spnego._asn1 import (
from spnego._context import GSSMech
from spnego._kerberos import (
from spnego._ntlm_raw.crypto import hmac_md5, ntowfv1, ntowfv2, rc4k
from spnego._ntlm_raw.messages import (
from spnego._spnego import InitialContextToken, NegTokenInit, NegTokenResp, unpack_token
from spnego._text import to_bytes
from spnego._tls_struct import (
def _parse_tls_handshake_server_key_exchange(view: memoryview, protocol_version: TlsProtocolVersion) -> typing.Dict[str, typing.Any]:
    curve_type = TlsECCurveType(struct.unpack('B', view[:1])[0])
    view = view[1:]
    curve = TlsSupportedGroup(struct.unpack('>H', view[:2])[0])
    view = view[2:]
    pubkey_len = struct.unpack('B', view[:1])[0]
    view = view[1:]
    pubkey = view[:pubkey_len].tobytes()
    view = view[pubkey_len:]
    signature_algo = None
    if protocol_version >= TlsProtocolVersion.tls1_2:
        signature_algo = TlsSignatureScheme(struct.unpack('>H', view[:2])[0])
        view = view[2:]
    signature_len = struct.unpack('>H', view[:2])[0]
    view = view[2:]
    signature = view[:signature_len].tobytes()
    return {'CurveType': parse_enum(curve_type), 'Curve': parse_enum(curve), 'PublicKey': base64.b16encode(pubkey).decode(), 'SignatureAlgorithm': parse_enum(signature_algo) if signature_algo else None, 'Signature': base64.b16encode(signature).decode()}