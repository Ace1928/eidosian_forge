from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class DataclassTypeError(PydanticTypeError):
    code = 'dataclass'
    msg_template = 'instance of {class_name}, tuple or dict expected'