from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class DecimalMaxDigitsError(PydanticValueError):
    code = 'decimal.max_digits'
    msg_template = 'ensure that there are no more than {max_digits} digits in total'

    def __init__(self, *, max_digits: int) -> None:
        super().__init__(max_digits=max_digits)