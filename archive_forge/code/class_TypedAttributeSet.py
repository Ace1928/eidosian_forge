from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import Any, TypeVar, final, overload
from ._exceptions import TypedAttributeLookupError
class TypedAttributeSet:
    """
    Superclass for typed attribute collections.

    Checks that every public attribute of every subclass has a type annotation.
    """

    def __init_subclass__(cls) -> None:
        annotations: dict[str, Any] = getattr(cls, '__annotations__', {})
        for attrname in dir(cls):
            if not attrname.startswith('_') and attrname not in annotations:
                raise TypeError(f'Attribute {attrname!r} is missing its type annotation')
        super().__init_subclass__()