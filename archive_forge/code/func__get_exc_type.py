import json
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Sequence, Tuple, Type, Union
from .json import pydantic_encoder
from .utils import Representation
def _get_exc_type(cls: Type[Exception]) -> str:
    if issubclass(cls, AssertionError):
        return 'assertion_error'
    base_name = 'type_error' if issubclass(cls, TypeError) else 'value_error'
    if cls in (TypeError, ValueError):
        return base_name
    code = getattr(cls, 'code', None) or cls.__name__.replace('Error', '').lower()
    return base_name + '.' + code