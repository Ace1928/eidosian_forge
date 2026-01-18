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
def _pattern_schema(self, pattern_type: Any) -> core_schema.CoreSchema:
    from . import _validators
    metadata = build_metadata_dict(js_functions=[lambda _1, _2: {'type': 'string', 'format': 'regex'}])
    ser = core_schema.plain_serializer_function_ser_schema(attrgetter('pattern'), when_used='json', return_schema=core_schema.str_schema())
    if pattern_type == typing.Pattern or pattern_type == re.Pattern:
        return core_schema.no_info_plain_validator_function(_validators.pattern_either_validator, serialization=ser, metadata=metadata)
    param = self._get_args_resolving_forward_refs(pattern_type, required=True)[0]
    if param == str:
        return core_schema.no_info_plain_validator_function(_validators.pattern_str_validator, serialization=ser, metadata=metadata)
    elif param == bytes:
        return core_schema.no_info_plain_validator_function(_validators.pattern_bytes_validator, serialization=ser, metadata=metadata)
    else:
        raise PydanticSchemaGenerationError(f'Unable to generate pydantic-core schema for {pattern_type!r}.')