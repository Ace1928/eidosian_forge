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
def _model_schema(self, cls: type[BaseModel]) -> core_schema.CoreSchema:
    """Generate schema for a Pydantic model."""
    with self.defs.get_schema_or_ref(cls) as (model_ref, maybe_schema):
        if maybe_schema is not None:
            return maybe_schema
        fields = cls.model_fields
        decorators = cls.__pydantic_decorators__
        computed_fields = decorators.computed_fields
        check_decorator_fields_exist(chain(decorators.field_validators.values(), decorators.field_serializers.values(), decorators.validators.values()), {*fields.keys(), *computed_fields.keys()})
        config_wrapper = ConfigWrapper(cls.model_config, check=False)
        core_config = config_wrapper.core_config(cls)
        metadata = build_metadata_dict(js_functions=[partial(modify_model_json_schema, cls=cls)])
        model_validators = decorators.model_validators.values()
        extras_schema = None
        if core_config.get('extra_fields_behavior') == 'allow':
            for tp in (cls, *cls.__mro__):
                extras_annotation = cls.__annotations__.get('__pydantic_extra__', None)
                if extras_annotation is not None:
                    tp = get_origin(extras_annotation)
                    if tp not in (Dict, dict):
                        raise PydanticSchemaGenerationError('The type annotation for `__pydantic_extra__` must be `Dict[str, ...]`')
                    extra_items_type = self._get_args_resolving_forward_refs(cls.__annotations__['__pydantic_extra__'], required=True)[1]
                    if extra_items_type is not Any:
                        extras_schema = self.generate_schema(extra_items_type)
                        break
        with self._config_wrapper_stack.push(config_wrapper), self._types_namespace_stack.push(cls):
            self = self._current_generate_schema
            if cls.__pydantic_root_model__:
                root_field = self._common_field_schema('root', fields['root'], decorators)
                inner_schema = root_field['schema']
                inner_schema = apply_model_validators(inner_schema, model_validators, 'inner')
                model_schema = core_schema.model_schema(cls, inner_schema, custom_init=getattr(cls, '__pydantic_custom_init__', None), root_model=True, post_init=getattr(cls, '__pydantic_post_init__', None), config=core_config, ref=model_ref, metadata=metadata)
            else:
                fields_schema: core_schema.CoreSchema = core_schema.model_fields_schema({k: self._generate_md_field_schema(k, v, decorators) for k, v in fields.items()}, computed_fields=[self._computed_field_schema(d, decorators.field_serializers) for d in computed_fields.values()], extras_schema=extras_schema, model_name=cls.__name__)
                inner_schema = apply_validators(fields_schema, decorators.root_validators.values(), None)
                new_inner_schema = define_expected_missing_refs(inner_schema, recursively_defined_type_refs())
                if new_inner_schema is not None:
                    inner_schema = new_inner_schema
                inner_schema = apply_model_validators(inner_schema, model_validators, 'inner')
                model_schema = core_schema.model_schema(cls, inner_schema, custom_init=getattr(cls, '__pydantic_custom_init__', None), root_model=False, post_init=getattr(cls, '__pydantic_post_init__', None), config=core_config, ref=model_ref, metadata=metadata)
            schema = self._apply_model_serializers(model_schema, decorators.model_serializers.values())
            schema = apply_model_validators(schema, model_validators, 'outer')
            self.defs.definitions[model_ref] = self._post_process_generated_schema(schema)
            return core_schema.definition_reference_schema(model_ref)