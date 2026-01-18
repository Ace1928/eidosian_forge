from __future__ import annotations
from typing import Any, List, Tuple, Union, Mapping, TypeVar
from urllib.parse import parse_qs, urlencode
from typing_extensions import Literal, get_args
from ._types import NOT_GIVEN, NotGiven, NotGivenOr
from ._utils import flatten
def _primitive_value_to_str(self, value: PrimitiveData) -> str:
    if value is True:
        return 'true'
    elif value is False:
        return 'false'
    elif value is None:
        return ''
    return str(value)