from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class NumberNotLeError(_NumberBoundError):
    code = 'number.not_le'
    msg_template = 'ensure this value is less than or equal to {limit_value}'