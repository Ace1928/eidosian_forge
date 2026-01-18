import typing
from spnego._asn1 import (
class TSCspDataDetail:
    """CredSSP TSCspDataDetail structure.

    The TSCspDataDetail structure contains CSP information used during smart card logon.

    The ASN.1 definition for the TSCspDataDetail structure is defined in `MS-CSSP 2.2.1.2.2.1 TSCspDataDetail`_::

        TSCspDataDetail ::= SEQUENCE {
                keySpec       [0] INTEGER,
                cardName      [1] OCTET STRING OPTIONAL,
                readerName    [2] OCTET STRING OPTIONAL,
                containerName [3] OCTET STRING OPTIONAL,
                cspName       [4] OCTET STRING OPTIONAL
        }

    Args:
        key_spec: The specification of the user's smart card.
        card_name: The name of the smart card.
        reader_name: The name of the smart card reader.
        container_name: The name of the certificate container.
        csp_name: The name of the CSP.

    Attributes:
        key_spec (int): See args.
        card_name (Optional[str]): See args.
        reader_name (Optional[str]): See args.
        container_name (Optional[str]): See args.
        csp_name (Optional[str]): See args.

    .. _MS-CSSP 2.2.1.2.2.1 TSCspDataDetail:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-cssp/34ee27b3-5791-43bb-9201-076054b58123
    """

    def __init__(self, key_spec: int, card_name: typing.Optional[str]=None, reader_name: typing.Optional[str]=None, container_name: typing.Optional[str]=None, csp_name: typing.Optional[str]=None) -> None:
        self.key_spec = key_spec
        self.card_name = card_name
        self.reader_name = reader_name
        self.container_name = container_name
        self.csp_name = csp_name

    def pack(self) -> bytes:
        """Packs the TSCspDataDetail as a byte string."""
        elements = [pack_asn1(TagClass.context_specific, True, 0, pack_asn1_integer(self.key_spec))]
        value_map = [(1, self.card_name), (2, self.reader_name), (3, self.container_name), (4, self.csp_name)]
        for idx, value in value_map:
            if value:
                b_value = value.encode('utf-16-le')
                elements.append(pack_asn1(TagClass.context_specific, True, idx, pack_asn1_octet_string(b_value)))
        return pack_asn1_sequence(elements)

    @staticmethod
    def unpack(b_data: typing.Union[ASN1Value, bytes]) -> 'TSCspDataDetail':
        """Unpacks the TSCspDataDetail TLV value."""
        csp_data = unpack_sequence(b_data)
        key_spec = get_sequence_value(csp_data, 0, 'TSCspDataDetail', 'keySpec', unpack_asn1_integer)
        card_name = unpack_text_field(csp_data, 1, 'TSCspDataDetail', 'cardName', default=None)
        reader_name = unpack_text_field(csp_data, 2, 'TSCspDataDetail', 'readerName', default=None)
        container_name = unpack_text_field(csp_data, 3, 'TSCspDataDetail', 'containerName', default=None)
        csp_name = unpack_text_field(csp_data, 4, 'TSCspDataDetail', 'cspName', default=None)
        return TSCspDataDetail(key_spec, card_name, reader_name, container_name, csp_name)