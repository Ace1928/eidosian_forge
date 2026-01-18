import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
class KrbApRep(KerberosV5Msg):
    """The KRB_AP_REP message.

    The KRB_AP_REP is a response to an application request `KRB_AP_REQ`.

    The ASN.1 definition for the KRB_AP_REP structure is defined in `RFC 4120 5.5.2`_::

        AP-REP          ::= [APPLICATION 15] SEQUENCE {
            pvno            [0] INTEGER (5),
            msg-type        [1] INTEGER (15),
            enc-part        [2] EncryptedData -- EncAPRepPart
        }

    Args:
        sequence: The ASN.1 sequence value as a dict to unpack.

    Attributes:
        enc_part (EncryptedData): The encrypted authenticator.

    .. _RFC 4120 5.5.2:
        https://www.rfc-editor.org/rfc/rfc4120#section-5.5.2
    """
    MESSAGE_TYPE = KerberosMessageType.ap_rep
    PARSE_MAP = [('pvno', 'PVNO', ParseType.default), ('msg-type', 'MESSAGE_TYPE', ParseType.enum), ('enc-part', 'enc_part', ParseType.token)]

    def __init__(self, sequence: typing.Dict[int, ASN1Value]) -> None:
        self.enc_part = get_sequence_value(sequence, 2, 'AP-REP', 'enc-part', EncryptedData.unpack)