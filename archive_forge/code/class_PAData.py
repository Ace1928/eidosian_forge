import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
class PAData:
    """Kerberos PA-DATA.

    The ASN.1 definition for the PA-DATA structure is defined in `RFC 4120 5.2.7`_::

        PA-DATA         ::= SEQUENCE {
            -- NOTE: first tag is [1], not [0]
            padata-type     [1] Int32,
            padata-value    [2] OCTET STRING -- might be encoded AP-REQ
        }

    Args:
        data_type: Indicates the type of data the value represents.
        value: The PAData value, usually the DER encoding of another message.

    Attributes:
        data_type (Union[int, KerberosPADataType]): See args.
        b_value (bytes): The raw bytes of padata-value, use `value` to get a structured object of these bytes if
            available.

    .. RFC 4120 5.2.7:
        https://www.rfc-editor.org/rfc/rfc4120#section-5.2.7
    """
    PARSE_MAP = [('padata-type', 'data_type', ParseType.enum), ('padata-value', 'value', ParseType.token)]

    def __init__(self, data_type: typing.Union[int, KerberosPADataType], value: bytes) -> None:
        self.data_type = data_type
        self.b_value = value

    @property
    def value(self) -> typing.Any:
        if self.data_type == KerberosPADataType.tgs_req:
            return KrbTgsReq.unpack(unpack_asn1(unpack_asn1(self.b_value)[0].b_data)[0])
        data_type_map = {int(KerberosPADataType.enc_timestamp): (EncryptedData.unpack, False), int(KerberosPADataType.etype_info2): (PAETypeInfo2.unpack, True)}
        if self.data_type in data_type_map:
            unpack_func, is_sequence = data_type_map[int(self.data_type)]
            b_value = unpack_asn1(self.b_value)[0]
            if is_sequence:
                return [unpack_func(v) for v in unpack_asn1_sequence(b_value)]
            else:
                return unpack_func(b_value.b_data)
        else:
            return self.b_value

    @staticmethod
    def unpack(value: typing.Union[ASN1Value, bytes]) -> 'PAData':
        sequence = unpack_asn1_tagged_sequence(value)

        def unpack_data_type(value: typing.Union[ASN1Value, bytes]) -> typing.Union[KerberosPADataType, int]:
            int_val = unpack_asn1_integer(value)
            try:
                return KerberosPADataType(int_val)
            except ValueError:
                return int_val
        data_type = get_sequence_value(sequence, 1, 'PA-DATA', 'padata-type', unpack_data_type)
        pa_value = get_sequence_value(sequence, 2, 'PA-DATA', 'padata-value', unpack_asn1_octet_string)
        return PAData(data_type, pa_value)