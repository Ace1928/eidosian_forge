from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class EnumMemberError(PydanticTypeError):
    code = 'enum'

    def __str__(self) -> str:
        permitted = ', '.join((repr(v.value) for v in self.enum_values))
        return f'value is not a valid enumeration member; permitted: {permitted}'