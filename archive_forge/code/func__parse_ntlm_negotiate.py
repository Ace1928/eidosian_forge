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
def _parse_ntlm_negotiate(data: Negotiate) -> typing.Dict[str, typing.Any]:
    b_data = data.pack()
    msg = {'NegotiateFlags': parse_flags(data.flags, enum_type=NegotiateFlags), 'DomainNameFields': {'Len': struct.unpack('<H', b_data[16:18])[0], 'MaxLen': struct.unpack('<H', b_data[18:20])[0], 'BufferOffset': struct.unpack('<I', b_data[20:24])[0]}, 'WorkstationFields': {'Len': struct.unpack('<H', b_data[24:26])[0], 'MaxLen': struct.unpack('<H', b_data[26:28])[0], 'BufferOffset': struct.unpack('<I', b_data[28:32])[0]}, 'Version': _parse_ntlm_version(data.version), 'Payload': {'DomainName': data.domain_name, 'Workstation': data.workstation}}
    return msg