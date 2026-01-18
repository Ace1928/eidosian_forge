import enum
import typing
def _add_missing_enum_member(cls: typing.Type[enum.IntEnum], value: object, label: str) -> typing.Optional[enum.Enum]:
    if not isinstance(value, int):
        return None
    new_member = int.__new__(cls)
    new_member._name_ = label.format(value)
    new_member._value_ = value
    return cls._value2member_map_.setdefault(value, new_member)