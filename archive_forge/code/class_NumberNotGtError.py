from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class NumberNotGtError(_NumberBoundError):
    code = 'number.not_gt'
    msg_template = 'ensure this value is greater than {limit_value}'