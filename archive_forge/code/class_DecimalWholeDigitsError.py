from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class DecimalWholeDigitsError(PydanticValueError):
    code = 'decimal.whole_digits'
    msg_template = 'ensure that there are no more than {whole_digits} digits before the decimal point'

    def __init__(self, *, whole_digits: int) -> None:
        super().__init__(whole_digits=whole_digits)