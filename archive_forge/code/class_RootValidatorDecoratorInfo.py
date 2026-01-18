from __future__ import annotations as _annotations
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property, partial, partialmethod
from inspect import Parameter, Signature, isdatadescriptor, ismethoddescriptor, signature
from itertools import islice
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Iterable, TypeVar, Union
from pydantic_core import PydanticUndefined, core_schema
from typing_extensions import Literal, TypeAlias, is_typeddict
from ..errors import PydanticUserError
from ._core_utils import get_type_ref
from ._internal_dataclass import slots_true
from ._typing_extra import get_function_type_hints
@dataclass(**slots_true)
class RootValidatorDecoratorInfo:
    """A container for data from `@root_validator` so that we can access it
    while building the pydantic-core schema.

    Attributes:
        decorator_repr: A class variable representing the decorator string, '@root_validator'.
        mode: The proposed validator mode.
    """
    decorator_repr: ClassVar[str] = '@root_validator'
    mode: Literal['before', 'after']