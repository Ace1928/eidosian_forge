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
def _match_generic_type(self, obj: Any, origin: Any) -> CoreSchema:
    if isinstance(origin, TypeAliasType):
        return self._type_alias_type_schema(obj)
    if _typing_extra.is_dataclass(origin):
        return self._dataclass_schema(obj, origin)
    if _typing_extra.is_namedtuple(origin):
        return self._namedtuple_schema(obj, origin)
    from_property = self._generate_schema_from_property(origin, obj)
    if from_property is not None:
        return from_property
    if _typing_extra.origin_is_union(origin):
        return self._union_schema(obj)
    elif origin in TUPLE_TYPES:
        return self._tuple_schema(obj)
    elif origin in LIST_TYPES:
        return self._list_schema(obj, self._get_first_arg_or_any(obj))
    elif origin in SET_TYPES:
        return self._set_schema(obj, self._get_first_arg_or_any(obj))
    elif origin in FROZEN_SET_TYPES:
        return self._frozenset_schema(obj, self._get_first_arg_or_any(obj))
    elif origin in DICT_TYPES:
        return self._dict_schema(obj, *self._get_first_two_args_or_any(obj))
    elif is_typeddict(origin):
        return self._typed_dict_schema(obj, origin)
    elif origin in (typing.Type, type):
        return self._subclass_schema(obj)
    elif origin in {typing.Sequence, collections.abc.Sequence}:
        return self._sequence_schema(obj)
    elif origin in {typing.Iterable, collections.abc.Iterable, typing.Generator, collections.abc.Generator}:
        return self._iterable_schema(obj)
    elif origin in (re.Pattern, typing.Pattern):
        return self._pattern_schema(obj)
    if self._arbitrary_types:
        return self._arbitrary_type_schema(origin)
    return self._unknown_type_schema(obj)