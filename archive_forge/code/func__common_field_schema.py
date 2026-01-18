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
def _common_field_schema(self, name: str, field_info: FieldInfo, decorators: DecoratorInfos) -> _CommonField:
    from .. import AliasChoices, AliasPath
    from ..fields import FieldInfo
    if has_instance_in_type(field_info.annotation, (ForwardRef, str)):
        types_namespace = self._types_namespace
        if self._typevars_map:
            types_namespace = (types_namespace or {}).copy()
            types_namespace.update({k.__name__: v for k, v in self._typevars_map.items()})
        evaluated = _typing_extra.eval_type_lenient(field_info.annotation, types_namespace)
        if evaluated is not field_info.annotation and (not has_instance_in_type(evaluated, PydanticRecursiveRef)):
            new_field_info = FieldInfo.from_annotation(evaluated)
            field_info.annotation = new_field_info.annotation
            for k, v in new_field_info._attributes_set.items():
                if k not in field_info._attributes_set and k not in field_info.metadata_lookup:
                    setattr(field_info, k, v)
            field_info.metadata = [*new_field_info.metadata, *field_info.metadata]
    source_type, annotations = (field_info.annotation, field_info.metadata)

    def set_discriminator(schema: CoreSchema) -> CoreSchema:
        schema = self._apply_discriminator_to_union(schema, field_info.discriminator)
        return schema
    with self.field_name_stack.push(name):
        if field_info.discriminator is not None:
            schema = self._apply_annotations(source_type, annotations, transform_inner_schema=set_discriminator)
        else:
            schema = self._apply_annotations(source_type, annotations)
    this_field_validators = filter_field_decorator_info_by_field(decorators.validators.values(), name)
    if _validators_require_validate_default(this_field_validators):
        field_info.validate_default = True
    each_item_validators = [v for v in this_field_validators if v.info.each_item is True]
    this_field_validators = [v for v in this_field_validators if v not in each_item_validators]
    schema = apply_each_item_validators(schema, each_item_validators, name)
    schema = apply_validators(schema, filter_field_decorator_info_by_field(this_field_validators, name), name)
    schema = apply_validators(schema, filter_field_decorator_info_by_field(decorators.field_validators.values(), name), name)
    if not field_info.is_required():
        schema = wrap_default(field_info, schema)
    schema = self._apply_field_serializers(schema, filter_field_decorator_info_by_field(decorators.field_serializers.values(), name))
    json_schema_updates = {'title': field_info.title, 'description': field_info.description, 'examples': to_jsonable_python(field_info.examples)}
    json_schema_updates = {k: v for k, v in json_schema_updates.items() if v is not None}
    json_schema_extra = field_info.json_schema_extra
    metadata = build_metadata_dict(js_annotation_functions=[get_json_schema_update_func(json_schema_updates, json_schema_extra)])
    alias_generator = self._config_wrapper.alias_generator
    if alias_generator is not None:
        self._apply_alias_generator_to_field_info(alias_generator, field_info, name)
    if isinstance(field_info.validation_alias, (AliasChoices, AliasPath)):
        validation_alias = field_info.validation_alias.convert_to_aliases()
    else:
        validation_alias = field_info.validation_alias
    return _common_field(schema, serialization_exclude=True if field_info.exclude else None, validation_alias=validation_alias, serialization_alias=field_info.serialization_alias, frozen=field_info.frozen, metadata=metadata)