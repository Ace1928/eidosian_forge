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
def _type_alias_type_schema(self, obj: Any) -> CoreSchema:
    with self.defs.get_schema_or_ref(obj) as (ref, maybe_schema):
        if maybe_schema is not None:
            return maybe_schema
        origin = get_origin(obj) or obj
        annotation = origin.__value__
        typevars_map = get_standard_typevars_map(obj)
        with self._types_namespace_stack.push(origin):
            annotation = _typing_extra.eval_type_lenient(annotation, self._types_namespace)
            annotation = replace_types(annotation, typevars_map)
            schema = self.generate_schema(annotation)
            assert schema['type'] != 'definitions'
            schema['ref'] = ref
        self.defs.definitions[ref] = schema
        return core_schema.definition_reference_schema(ref)