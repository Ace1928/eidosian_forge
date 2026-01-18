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
@staticmethod
def _apply_alias_generator_to_field_info(alias_generator: Callable[[str], str] | AliasGenerator, field_info: FieldInfo, field_name: str) -> None:
    """Apply an alias_generator to aliases on a FieldInfo instance if appropriate.

        Args:
            alias_generator: A callable that takes a string and returns a string, or an AliasGenerator instance.
            field_info: The FieldInfo instance to which the alias_generator is (maybe) applied.
            field_name: The name of the field from which to generate the alias.
        """
    if field_info.alias_priority is None or field_info.alias_priority <= 1 or field_info.alias is None or (field_info.validation_alias is None) or (field_info.serialization_alias is None):
        alias, validation_alias, serialization_alias = (None, None, None)
        if isinstance(alias_generator, AliasGenerator):
            alias, validation_alias, serialization_alias = alias_generator.generate_aliases(field_name)
        elif isinstance(alias_generator, Callable):
            alias = alias_generator(field_name)
            if not isinstance(alias, str):
                raise TypeError(f'alias_generator {alias_generator} must return str, not {alias.__class__}')
        if field_info.alias_priority is None or field_info.alias_priority <= 1:
            field_info.alias_priority = 1
        if field_info.alias_priority == 1:
            field_info.serialization_alias = serialization_alias or alias
            field_info.validation_alias = validation_alias or alias
            field_info.alias = alias
        if field_info.alias is None:
            field_info.alias = alias
        if field_info.serialization_alias is None:
            field_info.serialization_alias = serialization_alias or alias
        if field_info.validation_alias is None:
            field_info.validation_alias = validation_alias or alias