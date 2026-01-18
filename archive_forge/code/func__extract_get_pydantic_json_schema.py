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
def _extract_get_pydantic_json_schema(tp: Any, schema: CoreSchema) -> GetJsonSchemaFunction | None:
    """Extract `__get_pydantic_json_schema__` from a type, handling the deprecated `__modify_schema__`."""
    js_modify_function = getattr(tp, '__get_pydantic_json_schema__', None)
    if hasattr(tp, '__modify_schema__'):
        from pydantic import BaseModel
        has_custom_v2_modify_js_func = js_modify_function is not None and BaseModel.__get_pydantic_json_schema__.__func__ not in (js_modify_function, getattr(js_modify_function, '__func__', None))
        if not has_custom_v2_modify_js_func:
            raise PydanticUserError('The `__modify_schema__` method is not supported in Pydantic v2. Use `__get_pydantic_json_schema__` instead.', code='custom-json-schema')
    if hasattr(tp, '__origin__') and (not isinstance(tp, type(Annotated[int, 'placeholder']))):
        return _extract_get_pydantic_json_schema(tp.__origin__, schema)
    if js_modify_function is None:
        return None
    return js_modify_function