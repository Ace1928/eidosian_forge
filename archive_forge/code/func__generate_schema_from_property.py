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
def _generate_schema_from_property(self, obj: Any, source: Any) -> core_schema.CoreSchema | None:
    """Try to generate schema from either the `__get_pydantic_core_schema__` function or
        `__pydantic_core_schema__` property.

        Note: `__get_pydantic_core_schema__` takes priority so it can
        decide whether to use a `__pydantic_core_schema__` attribute, or generate a fresh schema.
        """
    with self.defs.get_schema_or_ref(obj) as (_, maybe_schema):
        if maybe_schema is not None:
            return maybe_schema
    if obj is source:
        ref_mode = 'unpack'
    else:
        ref_mode = 'to-def'
    schema: CoreSchema
    get_schema = getattr(obj, '__get_pydantic_core_schema__', None)
    if get_schema is None:
        validators = getattr(obj, '__get_validators__', None)
        if validators is None:
            return None
        warn('`__get_validators__` is deprecated and will be removed, use `__get_pydantic_core_schema__` instead.', PydanticDeprecatedSince20)
        schema = core_schema.chain_schema([core_schema.with_info_plain_validator_function(v) for v in validators()])
    elif len(inspect.signature(get_schema).parameters) == 1:
        schema = get_schema(source)
    else:
        schema = get_schema(source, CallbackGetCoreSchemaHandler(self._generate_schema, self, ref_mode=ref_mode))
    schema = self._unpack_refs_defs(schema)
    if is_function_with_inner_schema(schema):
        ref = schema['schema'].pop('ref', None)
        if ref:
            schema['ref'] = ref
    else:
        ref = get_ref(schema)
    if ref:
        self.defs.definitions[ref] = self._post_process_generated_schema(schema)
        return core_schema.definition_reference_schema(ref)
    schema = self._post_process_generated_schema(schema)
    return schema