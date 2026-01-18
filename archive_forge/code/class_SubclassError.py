from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class SubclassError(PydanticTypeError):
    code = 'subclass'
    msg_template = 'subclass of {expected_class} expected'

    def __init__(self, *, expected_class: Type[Any]) -> None:
        super().__init__(expected_class=display_as_type(expected_class))