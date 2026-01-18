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
@staticmethod
def _validate_type_param(type_param: TypeOrInst[Property[Any]], *, help_allowed: bool=False) -> Property[Any]:
    if isinstance(type_param, type):
        if issubclass(type_param, Property):
            return type_param()
        else:
            type_param = type_param.__name__
    elif isinstance(type_param, Property):
        if type_param._help is not None and (not help_allowed):
            raise ValueError("setting 'help' on type parameters doesn't make sense")
        return type_param
    raise ValueError(f'expected a Property as type parameter, got {type_param}')