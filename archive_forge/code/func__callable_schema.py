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
def _callable_schema(self, function: Callable[..., Any]) -> core_schema.CallSchema:
    """Generate schema for a Callable.

        TODO support functional validators once we support them in Config
        """
    sig = signature(function)
    type_hints = _typing_extra.get_function_type_hints(function)
    mode_lookup: dict[_ParameterKind, Literal['positional_only', 'positional_or_keyword', 'keyword_only']] = {Parameter.POSITIONAL_ONLY: 'positional_only', Parameter.POSITIONAL_OR_KEYWORD: 'positional_or_keyword', Parameter.KEYWORD_ONLY: 'keyword_only'}
    arguments_list: list[core_schema.ArgumentsParameter] = []
    var_args_schema: core_schema.CoreSchema | None = None
    var_kwargs_schema: core_schema.CoreSchema | None = None
    for name, p in sig.parameters.items():
        if p.annotation is sig.empty:
            annotation = Any
        else:
            annotation = type_hints[name]
        parameter_mode = mode_lookup.get(p.kind)
        if parameter_mode is not None:
            arg_schema = self._generate_parameter_schema(name, annotation, p.default, parameter_mode)
            arguments_list.append(arg_schema)
        elif p.kind == Parameter.VAR_POSITIONAL:
            var_args_schema = self.generate_schema(annotation)
        else:
            assert p.kind == Parameter.VAR_KEYWORD, p.kind
            var_kwargs_schema = self.generate_schema(annotation)
    return_schema: core_schema.CoreSchema | None = None
    config_wrapper = self._config_wrapper
    if config_wrapper.validate_return:
        return_hint = type_hints.get('return')
        if return_hint is not None:
            return_schema = self.generate_schema(return_hint)
    return core_schema.call_schema(core_schema.arguments_schema(arguments_list, var_args_schema=var_args_schema, var_kwargs_schema=var_kwargs_schema, populate_by_name=config_wrapper.populate_by_name), function, return_schema=return_schema)