import typing
from spnego._asn1 import (
class TSRemoteGuardPackageCred:
    """CredSSP TSRemoteGuardPackageCred structure.

    The TSRemoteGuardPackageCred structure contains credentials for a specific security package.

    The ASN.1 definition for the TSRemoteGuardPackageCred structure is defined in
    `MS-CSSP 2.2.1.2.3.1 TSRemoteGuardPackageCred`_::

        TSRemoteGuardPackageCred ::= SEQUENCE{
            packageName [0] OCTET STRING,
            credBuffer  [1] OCTET STRING,
        }

    Args:
        package_name: The name of the packages for which these credentials are intended.
        cred_buffer: The credentials in a format specified by the CredSSP server.

    Attributes:
        package_name (str): See args.
        cred_buffer (bytes): See args.

    .. _MS-CSSP 2.2.1.2.3.1 TSRemoteGuardPackageCred:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-cssp/173eee44-1a2c-463f-b909-c15db01e68d7
    """

    def __init__(self, package_name: str, cred_buffer: bytes) -> None:
        self.package_name = package_name
        self.cred_buffer = cred_buffer

    def pack(self) -> bytes:
        """Packs the TSRemoteGuardPackageCred as a byte string."""
        b_package_name = self.package_name.encode('utf-16-le')
        return pack_asn1_sequence([pack_asn1(TagClass.context_specific, True, 0, pack_asn1_octet_string(b_package_name)), pack_asn1(TagClass.context_specific, True, 1, pack_asn1_octet_string(self.cred_buffer))])

    @staticmethod
    def unpack(b_data: typing.Union[ASN1Value, bytes]) -> 'TSRemoteGuardPackageCred':
        """Unpacks the TSRemoteGuardPackageCred TLV value."""
        package_cred = unpack_sequence(b_data)
        package_name = unpack_text_field(package_cred, 0, 'TSRemoteGuardPackageCred', 'packageName') or ''
        cred_buffer = get_sequence_value(package_cred, 1, 'TSRemoteGuardPackageCred', 'credBuffer', unpack_asn1_octet_string)
        return TSRemoteGuardPackageCred(package_name, cred_buffer)