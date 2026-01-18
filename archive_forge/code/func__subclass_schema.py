from __future__ import annotations as _annotations
import collections.abc
import dataclasses
import inspect
import re
import sys
import typing
import warnings
from contextlib import contextmanager
from copy import copy, deepcopy
from enum import Enum
from functools import partial
from inspect import Parameter, _ParameterKind, signature
from itertools import chain
from operator import attrgetter
from types import FunctionType, LambdaType, MethodType
from typing import (
from warnings import warn
from pydantic_core import CoreSchema, PydanticUndefined, core_schema, to_jsonable_python
from typing_extensions import Annotated, Literal, TypeAliasType, TypedDict, get_args, get_origin, is_typeddict
from ..aliases import AliasGenerator
from ..annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
from ..config import ConfigDict, JsonDict, JsonEncoder
from ..errors import PydanticSchemaGenerationError, PydanticUndefinedAnnotation, PydanticUserError
from ..json_schema import JsonSchemaValue
from ..version import version_short
from ..warnings import PydanticDeprecatedSince20
from . import _core_utils, _decorators, _discriminated_union, _known_annotated_metadata, _typing_extra
from ._config import ConfigWrapper, ConfigWrapperStack
from ._core_metadata import CoreMetadataHandler, build_metadata_dict
from ._core_utils import (
from ._decorators import (
from ._fields import collect_dataclass_fields, get_type_hints_infer_globalns
from ._forward_ref import PydanticRecursiveRef
from ._generics import get_standard_typevars_map, has_instance_in_type, recursively_defined_type_refs, replace_types
from ._schema_generation_shared import (
from ._typing_extra import is_finalvar
from ._utils import lenient_issubclass
def _subclass_schema(self, type_: Any) -> core_schema.CoreSchema:
    """Generate schema for a Type, e.g. `Type[int]`."""
    type_param = self._get_first_arg_or_any(type_)
    if type_param == Any:
        return self._type_schema()
    elif isinstance(type_param, typing.TypeVar):
        if type_param.__bound__:
            if _typing_extra.origin_is_union(get_origin(type_param.__bound__)):
                return self._union_is_subclass_schema(type_param.__bound__)
            return core_schema.is_subclass_schema(type_param.__bound__)
        elif type_param.__constraints__:
            return core_schema.union_schema([self.generate_schema(typing.Type[c]) for c in type_param.__constraints__])
        else:
            return self._type_schema()
    elif _typing_extra.origin_is_union(get_origin(type_param)):
        return self._union_is_subclass_schema(type_param)
    else:
        return core_schema.is_subclass_schema(type_param)