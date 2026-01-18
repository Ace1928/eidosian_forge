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
def _dataclass_schema(self, dataclass: type[StandardDataclass], origin: type[StandardDataclass] | None) -> core_schema.CoreSchema:
    """Generate schema for a dataclass."""
    with self.defs.get_schema_or_ref(dataclass) as (dataclass_ref, maybe_schema):
        if maybe_schema is not None:
            return maybe_schema
        typevars_map = get_standard_typevars_map(dataclass)
        if origin is not None:
            dataclass = origin
        config = getattr(dataclass, '__pydantic_config__', None)
        with self._config_wrapper_stack.push(config), self._types_namespace_stack.push(dataclass):
            core_config = self._config_wrapper.core_config(dataclass)
            self = self._current_generate_schema
            from ..dataclasses import is_pydantic_dataclass
            if is_pydantic_dataclass(dataclass):
                fields = deepcopy(dataclass.__pydantic_fields__)
                if typevars_map:
                    for field in fields.values():
                        field.apply_typevars_map(typevars_map, self._types_namespace)
            else:
                fields = collect_dataclass_fields(dataclass, self._types_namespace, typevars_map=typevars_map)
            if config and config.get('extra') == 'allow':
                for field_name, field in fields.items():
                    if field.init is False:
                        raise PydanticUserError(f'Field {field_name} has `init=False` and dataclass has config setting `extra="allow"`. This combination is not allowed.', code='dataclass-init-false-extra-allow')
            decorators = dataclass.__dict__.get('__pydantic_decorators__') or DecoratorInfos.build(dataclass)
            args = sorted((self._generate_dc_field_schema(k, v, decorators) for k, v in fields.items()), key=lambda a: a.get('kw_only') is not False)
            has_post_init = hasattr(dataclass, '__post_init__')
            has_slots = hasattr(dataclass, '__slots__')
            args_schema = core_schema.dataclass_args_schema(dataclass.__name__, args, computed_fields=[self._computed_field_schema(d, decorators.field_serializers) for d in decorators.computed_fields.values()], collect_init_only=has_post_init)
            inner_schema = apply_validators(args_schema, decorators.root_validators.values(), None)
            model_validators = decorators.model_validators.values()
            inner_schema = apply_model_validators(inner_schema, model_validators, 'inner')
            dc_schema = core_schema.dataclass_schema(dataclass, inner_schema, post_init=has_post_init, ref=dataclass_ref, fields=[field.name for field in dataclasses.fields(dataclass)], slots=has_slots, config=core_config)
            schema = self._apply_model_serializers(dc_schema, decorators.model_serializers.values())
            schema = apply_model_validators(schema, model_validators, 'outer')
            self.defs.definitions[dataclass_ref] = self._post_process_generated_schema(schema)
            return core_schema.definition_reference_schema(dataclass_ref)