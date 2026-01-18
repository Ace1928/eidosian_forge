from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class PyObjectError(PydanticTypeError):
    msg_template = 'ensure this value contains valid import path or valid callable: {error_message}'