from __future__ import annotations
import logging # isort:skip
from copy import copy
from typing import (
from ...util.dependencies import uses_pandas
from ...util.strings import nice_join
from ..has_props import HasProps
from ._sphinx import property_link, register_type_link, type_link
from .descriptor_factory import PropertyDescriptorFactory
from .descriptors import PropertyDescriptor
from .singletons import (
class SingleParameterizedProperty(ParameterizedProperty[T]):
    """ A parameterized property with a single type parameter. """

    @property
    def type_param(self) -> Property[T]:
        return self._type_params[0]

    def __init__(self, type_param: TypeOrInst[Property[Any]], *, default: Init[T]=Intrinsic, help: str | None=None):
        super().__init__(type_param, default=default, help=help)

    def validate(self, value: Any, detail: bool=True) -> None:
        super().validate(value, detail=detail)
        self.type_param.validate(value, detail=detail)

    def transform(self, value: T) -> T:
        return self.type_param.transform(value)

    def wrap(self, value: T) -> T:
        return self.type_param.wrap(value)