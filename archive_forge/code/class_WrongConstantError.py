from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class WrongConstantError(PydanticValueError):
    code = 'const'

    def __str__(self) -> str:
        permitted = ', '.join((repr(v) for v in self.permitted))
        return f'unexpected value; permitted: {permitted}'