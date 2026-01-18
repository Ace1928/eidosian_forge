from __future__ import annotations
import collections
import collections.abc
import dataclasses
import enum
import functools
import inspect
import itertools
import numbers
import os
import sys
import typing
import warnings
from typing import (
import docstring_parser
import typing_extensions
from typing_extensions import (
from . import (
from ._typing import TypeForm
from .conf import _confstruct, _markers
def _field_list_from_pydantic(cls: TypeForm[Any], default_instance: DefaultInstance) -> Union[List[FieldDefinition], UnsupportedNestedTypeMessage]:
    assert pydantic is not None
    field_list = []
    pydantic_version = int(getattr(pydantic, '__version__', '1.0.0').partition('.')[0])
    if pydantic_version < 2 or (pydantic_v1 is not None and issubclass(cls, pydantic_v1.BaseModel)):
        if TYPE_CHECKING:
            cls_cast = cast(pydantic_v1.BaseModel, cls)
        else:
            cls_cast = cls
        for pd1_field in cls_cast.__fields__.values():
            helptext = pd1_field.field_info.description
            if helptext is None:
                helptext = _docstrings.get_field_docstring(cls, pd1_field.name)
            default, is_default_from_default_instance = _get_pydantic_v1_field_default(pd1_field.name, pd1_field, default_instance)
            field_list.append(FieldDefinition.make(name=pd1_field.name, type_or_callable=pd1_field.outer_type_, default=default, is_default_from_default_instance=is_default_from_default_instance, helptext=helptext))
    else:
        for name, pd2_field in cast(pydantic.BaseModel, cls).model_fields.items():
            helptext = pd2_field.description
            if helptext is None:
                helptext = _docstrings.get_field_docstring(cls, name)
            default, is_default_from_default_instance = _get_pydantic_v2_field_default(name, pd2_field, default_instance)
            field_list.append(FieldDefinition.make(name=name, type_or_callable=Annotated.__class_getitem__((pd2_field.annotation,) + tuple(pd2_field.metadata)) if len(pd2_field.metadata) > 0 else pd2_field.annotation, default=default, is_default_from_default_instance=is_default_from_default_instance, helptext=helptext))
    return field_list