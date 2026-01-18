from __future__ import annotations
import logging # isort:skip
from typing import Any, TypeVar
from ._sphinx import property_link, register_type_link, type_link
from .bases import (
from .singletons import Intrinsic
class NotSerialized(SingleParameterizedProperty[T]):
    """
    A property which state won't be synced with the browser.
    """
    _serialized = False

    def __init__(self, type_param: TypeOrInst[Property[T]], *, default: Init[T]=Intrinsic, help: str | None=None) -> None:
        super().__init__(type_param, default=default, help=help)