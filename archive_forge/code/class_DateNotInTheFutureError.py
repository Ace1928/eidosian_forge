from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class DateNotInTheFutureError(PydanticValueError):
    code = 'date.not_in_the_future'
    msg_template = 'date is not in the future'