import enum
import typing
class TlsECCurveType(enum.IntEnum):
    unassigned = 0
    explicit_primve = 1
    explicit_char2 = 2
    named_curve = 3

    @classmethod
    def _missing_(cls, value: object) -> typing.Optional[enum.Enum]:
        return _add_missing_enum_member(cls, value, 'Unknown EC Curve Type 0x{0:02X}')