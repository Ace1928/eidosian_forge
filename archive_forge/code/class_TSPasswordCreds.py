import typing
from spnego._asn1 import (
class TSPasswordCreds:
    """CredSSP TSPasswordCreds structure.

    The TSPasswordCreds structure contains the user's password credentials that are delegated to the server.

    The ASN.1 definition for the TSPasswordCreds structure is defined in `MS-CSSP 2.2.1.2.1 TSPasswordCreds`_::

        TSPasswordCreds ::= SEQUENCE {
                domainName  [0] OCTET STRING,
                userName    [1] OCTET STRING,
                password    [2] OCTET STRING
        }

    Args:
        domain_name: The name of the user's account domain.
        username: The user's account name.
        password: The user's account password.

    Attributes:
        domain_name (str): See args.
        username (str): See args.
        password (str): See args.

    .. _MS-CSSP 2.2.1.2.1 TSPasswordCreds:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-cssp/9664994d-0784-4659-b85b-83b8d54c2336
    """

    def __init__(self, domain_name: str, username: str, password: str) -> None:
        self.domain_name = domain_name
        self.username = username
        self.password = password

    def pack(self) -> bytes:
        """Packs the TSPasswordCreds as a byte string."""
        elements = []
        for idx, value in enumerate([self.domain_name, self.username, self.password]):
            b_value = value.encode('utf-16-le')
            elements.append(pack_asn1(TagClass.context_specific, True, idx, pack_asn1_octet_string(b_value)))
        return pack_asn1_sequence(elements)

    @staticmethod
    def unpack(b_data: typing.Union[ASN1Value, bytes]) -> 'TSPasswordCreds':
        """Unpacks the TSPasswordCreds TLV value."""
        creds = unpack_sequence(b_data)
        domain_name = unpack_text_field(creds, 0, 'TSPasswordCreds', 'domainName') or ''
        username = unpack_text_field(creds, 1, 'TSPasswordCreds', 'userName') or ''
        password = unpack_text_field(creds, 2, 'TSPasswordCreds', 'password') or ''
        return TSPasswordCreds(domain_name, username, password)