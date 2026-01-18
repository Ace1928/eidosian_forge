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
def _parse_ntlm_version(version: typing.Optional[Version]) -> typing.Optional[typing.Dict[str, typing.Union[int, str]]]:
    if not version:
        return None
    return {'Major': version.major, 'Minor': version.minor, 'Build': version.build, 'Reserved': base64.b16encode(version.reserved).decode(), 'NTLMRevision': version.revision}