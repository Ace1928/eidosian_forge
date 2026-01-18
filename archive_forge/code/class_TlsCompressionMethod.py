import enum
import typing
class TlsCompressionMethod(enum.IntEnum):
    none = 0
    deflate = 1
    lzs = 64

    @classmethod
    def _missing_(cls, value: object) -> typing.Optional[enum.Enum]:
        return _add_missing_enum_member(cls, value, 'Unknown Compression Method 0x{0:02X}')