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
def _union_schema(self, union_type: Any) -> core_schema.CoreSchema:
    """Generate schema for a Union."""
    args = self._get_args_resolving_forward_refs(union_type, required=True)
    choices: list[CoreSchema] = []
    nullable = False
    for arg in args:
        if arg is None or arg is _typing_extra.NoneType:
            nullable = True
        else:
            choices.append(self.generate_schema(arg))
    if len(choices) == 1:
        s = choices[0]
    else:
        choices_with_tags: list[CoreSchema | tuple[CoreSchema, str]] = []
        for choice in choices:
            metadata = choice.get('metadata')
            if isinstance(metadata, dict):
                tag = metadata.get(_core_utils.TAGGED_UNION_TAG_KEY)
                if tag is not None:
                    choices_with_tags.append((choice, tag))
                else:
                    choices_with_tags.append(choice)
        s = core_schema.union_schema(choices_with_tags)
    if nullable:
        s = core_schema.nullable_schema(s)
    return s