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
def _parse_spnego_init(data: NegTokenInit, secret: typing.Optional[str]=None, encoding: typing.Optional[str]=None) -> typing.Dict[str, typing.Any]:
    mech_types = [parse_enum(m, enum_type=GSSMech) for m in data.mech_types] if data.mech_types else None
    mech_token = None
    if data.mech_token:
        mech_token = parse_token(data.mech_token, secret=secret, encoding=encoding)
    encoding = encoding or 'utf-8'
    msg = {'mechTypes': mech_types, 'reqFlags': parse_flags(data.req_flags) if data.req_flags is not None else None, 'mechToken': mech_token, 'mechListMIC': base64.b16encode(data.mech_list_mic).decode() if data.mech_list_mic is not None else None}
    if data.hint_name or data.hint_address:
        msg['negHints'] = {'hintName': data.hint_name.decode(encoding) if data.hint_name else None, 'hintAddress': data.hint_address.decode(encoding) if data.hint_address else None}
    return msg