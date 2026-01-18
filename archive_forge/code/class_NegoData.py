import typing
from spnego._asn1 import (
class NegoData:
    """CredSSP NegoData structure.

    The NegoData structure contains the SPNEGO tokens, the Kerberos messages, or the NTLM messages. While the
    structure is a SEQUENCE OF SEQUENCE this class just represents an individual SEQUENCE entry.

    The ASN.1 definition for the NegoData structure is defined in `MS-CSSP 2.2.1.1 NegoData`_::

        NegoData ::= SEQUENCE OF SEQUENCE {
                negoToken [0] OCTET STRING
        }

    Args:
        nego_token: One or more SPNEGO tokens and all Kerberos or NTLM messages, as negotiated by SPNEGO.

    Attributes:
        nego_token (bytes): See args.

    .. _MS-CSSP 2.2.1.1 NegoData:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-cssp/9664994d-0784-4659-b85b-83b8d54c2336
    """

    def __init__(self, nego_token: bytes) -> None:
        self.nego_token = nego_token

    def pack(self) -> bytes:
        """Packs the NegoData as a byte string."""
        return pack_asn1_sequence([pack_asn1(TagClass.context_specific, True, 0, pack_asn1_octet_string(self.nego_token))])

    @staticmethod
    def unpack(b_data: typing.Union[ASN1Value, bytes]) -> 'NegoData':
        """Unpacks the NegoData TLV value."""
        nego_data = unpack_sequence(b_data)
        nego_token = get_sequence_value(nego_data, 0, 'NegoData', 'negoToken', unpack_asn1_octet_string)
        return NegoData(nego_token)