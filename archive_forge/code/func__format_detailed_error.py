from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union
from bson.errors import InvalidDocument
def _format_detailed_error(message: str, details: Optional[Union[Mapping[str, Any], list[Any]]]) -> str:
    if details is not None:
        message = f'{message}, full error: {details}'
    return message