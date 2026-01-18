from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class ArbitraryTypeError(PydanticTypeError):
    code = 'arbitrary_type'
    msg_template = 'instance of {expected_arbitrary_type} expected'

    def __init__(self, *, expected_arbitrary_type: Type[Any]) -> None:
        super().__init__(expected_arbitrary_type=display_as_type(expected_arbitrary_type))