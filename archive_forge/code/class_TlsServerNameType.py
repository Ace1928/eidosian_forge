import enum
import typing
class TlsServerNameType(enum.IntEnum):
    server_name = 0

    @classmethod
    def _missing_(cls, value: object) -> typing.Optional[enum.Enum]:
        return _add_missing_enum_member(cls, value, 'Unknown Server Name Type 0x{0:02X}')