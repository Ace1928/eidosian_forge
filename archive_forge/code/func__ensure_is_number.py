from typing import Any, Tuple, Union
from ._base import FloatObject, NumberObject
from ._data_structures import ArrayObject
def _ensure_is_number(self, value: Any) -> Union[FloatObject, NumberObject]:
    if not isinstance(value, (NumberObject, FloatObject)):
        value = FloatObject(value)
    return value