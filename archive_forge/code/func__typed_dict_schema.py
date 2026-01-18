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
def _typed_dict_schema(self, typed_dict_cls: Any, origin: Any) -> core_schema.CoreSchema:
    """Generate schema for a TypedDict.

        It is not possible to track required/optional keys in TypedDict without __required_keys__
        since TypedDict.__new__ erases the base classes (it replaces them with just `dict`)
        and thus we can track usage of total=True/False
        __required_keys__ was added in Python 3.9
        (https://github.com/miss-islington/cpython/blob/1e9939657dd1f8eb9f596f77c1084d2d351172fc/Doc/library/typing.rst?plain=1#L1546-L1548)
        however it is buggy
        (https://github.com/python/typing_extensions/blob/ac52ac5f2cb0e00e7988bae1e2a1b8257ac88d6d/src/typing_extensions.py#L657-L666).

        On 3.11 but < 3.12 TypedDict does not preserve inheritance information.

        Hence to avoid creating validators that do not do what users expect we only
        support typing.TypedDict on Python >= 3.12 or typing_extension.TypedDict on all versions
        """
    from ..fields import FieldInfo
    with self.defs.get_schema_or_ref(typed_dict_cls) as (typed_dict_ref, maybe_schema):
        if maybe_schema is not None:
            return maybe_schema
        typevars_map = get_standard_typevars_map(typed_dict_cls)
        if origin is not None:
            typed_dict_cls = origin
        if not _SUPPORTS_TYPEDDICT and type(typed_dict_cls).__module__ == 'typing':
            raise PydanticUserError('Please use `typing_extensions.TypedDict` instead of `typing.TypedDict` on Python < 3.12.', code='typed-dict-version')
        try:
            config: ConfigDict | None = get_attribute_from_bases(typed_dict_cls, '__pydantic_config__')
        except AttributeError:
            config = None
        with self._config_wrapper_stack.push(config), self._types_namespace_stack.push(typed_dict_cls):
            core_config = self._config_wrapper.core_config(typed_dict_cls)
            self = self._current_generate_schema
            required_keys: frozenset[str] = typed_dict_cls.__required_keys__
            fields: dict[str, core_schema.TypedDictField] = {}
            decorators = DecoratorInfos.build(typed_dict_cls)
            for field_name, annotation in get_type_hints_infer_globalns(typed_dict_cls, localns=self._types_namespace, include_extras=True).items():
                annotation = replace_types(annotation, typevars_map)
                required = field_name in required_keys
                if get_origin(annotation) == _typing_extra.Required:
                    required = True
                    annotation = self._get_args_resolving_forward_refs(annotation, required=True)[0]
                elif get_origin(annotation) == _typing_extra.NotRequired:
                    required = False
                    annotation = self._get_args_resolving_forward_refs(annotation, required=True)[0]
                field_info = FieldInfo.from_annotation(annotation)
                fields[field_name] = self._generate_td_field_schema(field_name, field_info, decorators, required=required)
            metadata = build_metadata_dict(js_functions=[partial(modify_model_json_schema, cls=typed_dict_cls)], typed_dict_cls=typed_dict_cls)
            td_schema = core_schema.typed_dict_schema(fields, computed_fields=[self._computed_field_schema(d, decorators.field_serializers) for d in decorators.computed_fields.values()], ref=typed_dict_ref, metadata=metadata, config=core_config)
            schema = self._apply_model_serializers(td_schema, decorators.model_serializers.values())
            schema = apply_model_validators(schema, decorators.model_validators.values(), 'all')
            self.defs.definitions[typed_dict_ref] = self._post_process_generated_schema(schema)
            return core_schema.definition_reference_schema(typed_dict_ref)