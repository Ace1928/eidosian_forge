import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
class KrbApReq(KerberosV5Msg):
    """The KRB_AP_REQ message.

    The KRB_AP_REQ message contains is used to authenticate the initiator to an acceptor.

    The ASN.1 definition for the KRB_AP_REQ structure is defined in `RFC 4120 5.5.1`_::

        AP-REQ          ::= [APPLICATION 14] SEQUENCE {
            pvno            [0] INTEGER (5),
            msg-type        [1] INTEGER (14),
            ap-options      [2] APOptions,
            ticket          [3] Ticket,
            authenticator   [4] EncryptedData -- Authenticator
        }

    Args:
        sequence: The ASN.1 sequence value as a dict to unpack.

    Attributes:
        ap_options (KerberosAPOptions): Options related to the AP request.
        ticket (Ticket): The ticket authenticating the client to the server.
        authenticator (EncryptedData): The encrypted authenticator.

    .. _RFC 4120 5.5.1:
        https://www.rfc-editor.org/rfc/rfc4120#section-5.5.1
    """
    MESSAGE_TYPE = KerberosMessageType.ap_req
    PARSE_MAP = [('pvno', 'PVNO', ParseType.default), ('msg-type', 'MESSAGE_TYPE', ParseType.enum), ('ap-options', 'ap_options', ParseType.flags), ('ticket', 'ticket', ParseType.token), ('authenticator', 'authenticator', ParseType.token)]

    def __init__(self, sequence: typing.Dict[int, ASN1Value]) -> None:
        raw_ap_options = get_sequence_value(sequence, 2, 'AP-REQ', 'ap-options', unpack_asn1_bit_string)
        ap_options = KerberosAPOptions(struct.unpack('<I', raw_ap_options)[0])
        self.ap_options = ap_options
        self.ticket = get_sequence_value(sequence, 3, 'AP-REQ', 'ticket', Ticket.unpack)
        self.authenticator = get_sequence_value(sequence, 4, 'AP-REQ', 'authenticator', EncryptedData.unpack)