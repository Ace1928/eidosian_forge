import enum
import typing
class TlsContentType(enum.IntEnum):
    invalid = 0
    change_cipher_spec = 20
    alert = 21
    handshake = 22
    application_data = 23

    @classmethod
    def _missing_(cls, value: object) -> typing.Optional[enum.Enum]:
        return _add_missing_enum_member(cls, value, 'Unknown TLS Content Type 0x{0:02X}')