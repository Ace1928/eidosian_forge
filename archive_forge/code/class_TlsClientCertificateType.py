import enum
import typing
class TlsClientCertificateType(enum.IntEnum):
    rsa_sign = 1
    dss_sign = 2
    rsa_fixed_dh = 3
    dss_fixed_dh = 4
    rsa_ephemeral_dh = 5
    dss_ephemeral_dh = 6
    fortezza_dms = 20
    ecdsa_sign = 64
    rsa_fixed_ecdh = 65
    ecdsa_fixed_ecdh = 66
    gost_sign256 = 67
    gost_sign512 = 68

    @classmethod
    def _missing_(cls, value: object) -> typing.Optional[enum.Enum]:
        return _add_missing_enum_member(cls, value, 'Unknown Client Certificate Type 0x{0:02X}')