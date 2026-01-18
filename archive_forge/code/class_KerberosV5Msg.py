import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
class KerberosV5Msg(metaclass=_KerberosMsgType):
    MESSAGE_TYPE = KerberosMessageType.unknown
    PVNO = 5

    def __init__(self, sequence: typing.Dict[int, ASN1Value]) -> None:
        self.sequence = sequence

    @staticmethod
    def unpack(value: typing.Union[ASN1Value, bytes]) -> 'KerberosV5Msg':
        msg_sequence = unpack_asn1_tagged_sequence(value)
        return KerberosV5Msg(msg_sequence)