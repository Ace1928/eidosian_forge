from __future__ import annotations
import logging # isort:skip
from typing import Any, TypeVar, Union
from ...util.deprecation import deprecated
from ._sphinx import property_link, register_type_link, type_link
from .bases import (
from .required import Required
from .singletons import Undefined
class NonNullable(Required[T]):
    """
    A property accepting a value of some other type while having undefined default.

    .. deprecated:: 3.0.0

        Use ``bokeh.core.property.required.Required`` instead.
    """

    def __init__(self, type_param: TypeOrInst[Property[T]], *, default: Init[T]=Undefined, help: str | None=None) -> None:
        deprecated((3, 0, 0), 'NonNullable(Type)', 'Required(Type)')
        super().__init__(type_param, default=default, help=help)