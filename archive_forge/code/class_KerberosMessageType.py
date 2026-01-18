import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
class KerberosMessageType(enum.IntEnum):
    unknown = 0
    as_req = 10
    as_rep = 11
    tgs_req = 12
    tgs_rep = 13
    ap_req = 14
    ap_rep = 15
    error = 30

    @classmethod
    def native_labels(cls) -> typing.Dict['KerberosMessageType', str]:
        return {KerberosMessageType.unknown: 'UNKNOWN', KerberosMessageType.as_req: 'AS-REQ', KerberosMessageType.as_rep: 'AS-REP', KerberosMessageType.tgs_req: 'TGS-REQ', KerberosMessageType.tgs_rep: 'TGS-REP', KerberosMessageType.ap_req: 'AP-REQ', KerberosMessageType.ap_rep: 'AP-REP', KerberosMessageType.error: 'KRB-ERROR'}