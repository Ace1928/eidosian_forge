from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class StrRegexError(PydanticValueError):
    code = 'str.regex'
    msg_template = 'string does not match regex "{pattern}"'

    def __init__(self, *, pattern: str) -> None:
        super().__init__(pattern=pattern)